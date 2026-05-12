import os

import ase.io
from ase import Atoms

from pygsm.level_of_theories.ase import ASELoT
from pygsm.potential_energy_surfaces import PES
from pygsm.growing_string_methods import DE_GSM
from pygsm.optimizers import eigenvector_follow, lbfgs
from pygsm.utilities import nifty
from pygsm.utilities.elements import ElementData
from pygsm.coordinate_systems import Topology, PrimitiveInternalCoordinates, DelocalizedInternalCoordinates
from pygsm.molecule import Molecule

from qc_launcher.drivers.base_driver import BaseDriver


def run_gsm(
    driver: BaseDriver,
    config: dict,
    atoms_list: list,
    filename: str = "molecule",
) -> None:
    # read config
    optimizer_method = config.get("optimizer_method", "eigenvector_follow")
    line_search = config.get("line_search", "NoLineSearch")
    climb = config.get("climb", True)
    dmax = config.get("dmax", 0.1)
    coordinate_type = config.get("coordinate_type", "TRIC")
    gsm_type = config.get("gsm_type", "DE_GSM")
    num_nodes = config.get("num_nodes", 11)
    add_node_tol = config.get("add_node_tol", 0.1)
    conv_tol = float(config.get("conv_tol", 5e-4))
    ediff = float(config.get("ediff", 100.0))
    fmax = float(config.get("fmax", 100.0))
    ID = config.get("ID", 0)
    max_gsm_steps = config.get("max_gsm_steps", 100)
    max_opt_steps = config.get("max_opt_steps", 3)
    fixed_reactant = config.get("fixed_reactant", False)
    fixed_product = config.get("fixed_product", False)

    # read atoms
    if gsm_type == "DE_GSM":
        assert len(atoms_list) == 2, "DE_GSM requires exactly two geometries (initial and final)."
        atoms_reactant: Atoms; atoms_product: Atoms
        atoms_reactant, atoms_product = atoms_list[0], atoms_list[1]
    elif gsm_type == "SE_GSM":
        assert len(atoms_list) == 1, "SE_GSM requires exactly one geometry (initial)."
        atoms_reactant: Atoms = atoms_list[0]
        raise NotImplementedError("SE_GSM is not implemented yet.")
    else:
        raise ValueError(f"Unsupported gsm_type: {gsm_type}. Supported types are 'DE_GSM' and 'SE_GSM'.")

    nifty.printcool("Parsed GSM")

    # set level of theory
    calc = driver.to_ase_calc()
    lot = ASELoT.from_options(
        calculator=calc,
        geom=[[x.symbol, *x.position] for x in atoms_reactant],
        ID=ID,
    )
    # build PES object
    multiplicity = atoms_reactant.info.get("multiplicity", 1)
    pes_obj = PES.from_options(lot=lot, ad_idx=0, multiplicity=multiplicity)

    # build topology
    nifty.printcool("Building topologies")
    element_table = ElementData()
    elements = [element_table.from_symbol(sym) for sym in atoms_reactant.get_chemical_symbols()]

    topology_reactant = Topology.build_topology(
        xyz=atoms_reactant.get_positions(),
        atoms=elements,
    )
    topology_product = Topology.build_topology(
        xyz=atoms_product.get_positions(),
        atoms=elements,
    )

    for bond in topology_product.edges():
        if bond in topology_reactant.edges() or (bond[1], bond[0]) in topology_reactant.edges():
            continue
        print(f"Adding bond {bond} to reactant topology")
        if bond[0] > bond[1]:
            topology_reactant.add_edge(bond[0], bond[1])
        else:
            topology_reactant.add_edge(bond[1], bond[0])
    
    # set internal coordinates
    nifty.printcool("Building Primitive Internal Coordinates")
    prim_reactant = PrimitiveInternalCoordinates.from_options(
        xyz=atoms_reactant.get_positions(),
        atoms=elements,
        topology=topology_reactant,
        connect=coordinate_type == "DLC",
        addtr=coordinate_type == "TRIC",
        addcart=coordinate_type == "HDLC",
    )
    prim_product = PrimitiveInternalCoordinates.from_options(
        xyz=atoms_product.get_positions(),
        atoms=elements,
        topology=topology_product,
        connect=coordinate_type == "DLC",
        addtr=coordinate_type == "TRIC",
        addcart=coordinate_type == "HDLC",
    )

    # add product coords to reactant coords
    prim_reactant.add_union_primitives(prim_product)

    # delocalized internal coordinates
    nifty.printcool("Building Delocalized Internal Coordinates")
    deloc_coords_reactant = DelocalizedInternalCoordinates.from_options(
        xyz=atoms_reactant.get_positions(),
        atoms=elements,
        connect=coordinate_type == "DLC",
        addtr=coordinate_type == "TRIC",
        addcart=coordinate_type == "HDLC",
        primitives=prim_reactant,
    )

    # molecules
    nifty.printcool(f"Building the reactant object with {coordinate_type}")
    form_hessian = optimizer_method == "eigenvector_follow"

    molecule_reactant = Molecule.from_options(
        geom=[[x.symbol, *x.position] for x in atoms_reactant],
        PES=pes_obj,
        coord_obj=deloc_coords_reactant,
        Form_Hessian=form_hessian,
    )
    molecule_product = Molecule.copy_from_options(
        molecule_reactant,
        xyz=atoms_product.get_positions(),
        new_node_id=num_nodes - 1,
        copy_wavefunction=False,
    )

    # optimizer
    nifty.printcool("Building optimizer")
    opt_options = dict(
        print_level=1,
        Linesearch=line_search,
        update_hess_in_bg=not (climb or optimizer_method == "lbfgs"),
        conv_Ediff=ediff,
        conv_gmax=fmax,
        DMAX=dmax,
        opt_climb=climb,
    )
    if optimizer_method == "eigenvector_follow":
        optimizer_object = eigenvector_follow.from_options(**opt_options)
    elif optimizer_method == "lbfgs":
        optimizer_object = lbfgs.from_options(**opt_options)
    else:
        raise ValueError(f"Unsupported optimizer_method: {optimizer_method}. Supported methods are 'eigenvector_follow' and 'lbfgs'.")
    
    # GSM
    nifty.printcool("Building GSM object")
    gsm = DE_GSM.from_options(
        reactant=molecule_reactant,
        product=molecule_product,
        nnodes=num_nodes,
        CONV_TOL=conv_tol,
        CONV_gmax=fmax,
        CONV_Ediff=ediff,
        ADD_NODE_TOL=add_node_tol,
        growth_direction=0,
        optimizer=optimizer_object,
        ID=ID,
        print_level=1,
        mp_cores=1,
        interp_method="DLC",
    )

    # optimize reactant and product if not fixed
    if not fixed_reactant:
        nifty.printcool("Optimizing reactant")
        path = os.path.join(os.getcwd(), "scratch", f"{ID:03}", "0")
        optimizer_object.optimize(
            molecule=molecule_reactant,
            refE=molecule_reactant.energy,
            opt_steps=100,
            path=path,
        )
    if not fixed_product:
        nifty.printcool("Optimizing product")
        path = os.path.join(os.getcwd(), "scratch", f"{ID:03}", f"{num_nodes - 1}")
        optimizer_object.optimize(
            molecule=molecule_product,
            refE=molecule_product.energy,
            opt_steps=100,
            path=path,
        )
    
    # set rtype
    rtype = 1 if climb else 2

    # run GSM
    nifty.printcool("Running GSM")
    gsm.go_gsm(max_iters=max_gsm_steps, opt_steps=max_opt_steps, rtype=rtype)

    # write the results into an xyz file
    string_ase, ts_ase = gsm_to_ase_atoms(gsm)
    ase.io.write(f"{filename}_GSM.xyz", string_ase)
    ase.io.write(f"{filename}_TS.xyz", ts_ase)



def gsm_to_ase_atoms(gsm: DE_GSM):
    # string
    frames = []
    for energy, geom in zip(gsm.energies, gsm.geometries):
        at = Atoms(symbols=[x[0] for x in geom], positions=[x[1:4] for x in geom])
        at.info["energy"] = energy
        frames.append(at)

    # TS
    ts_geom = gsm.nodes[gsm.TSnode].geometry
    ts_atoms = Atoms(symbols=[x[0] for x in ts_geom], positions=[x[1:4] for x in ts_geom])

    return frames, ts_atoms
