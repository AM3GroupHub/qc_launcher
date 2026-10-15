from __future__ import annotations

from typing import Any

import ase.io
from ase import Atoms
from pygsm import (
    DEGSM,
    LBFGS,
    PES,
    ASELoT,
    DelocalizedInternalCoordinates,
    EigenvectorFollow,
    Molecule,
    PrimitiveInternalCoordinates,
    build_topology,
)
from pygsm.utils import output
from pygsm.utils.elements import ElementData

from qc_launcher.drivers.base_driver import BaseDriver

# go_gsm rtype for each `ts_search` setting
TS_SEARCH_RTYPE = {"none": 0, "climb": 1, "exact": 2}
OPTIMIZERS = {"eigenvector_follow": EigenvectorFollow, "lbfgs": LBFGS}


def resolve_ts_search(config: dict) -> str:
    """`ts_search` (none | climb | exact), or the legacy `climb` flag.

    Legacy mapping, kept for existing configs: `climb: true` -> climbing image
    only, `climb: false` -> climbing image plus exact TS search.
    """
    ts_search = config.get("ts_search")
    if ts_search is None:
        return "climb" if config.get("climb", True) else "exact"
    if ts_search not in TS_SEARCH_RTYPE:
        raise ValueError(f"Unsupported ts_search: {ts_search}. Supported values are {list(TS_SEARCH_RTYPE)}.")
    return ts_search


def run_gsm(
    driver: BaseDriver,
    config: dict,
    atoms_list: list,
    filename: str = "molecule",
) -> None:

    # read config
    optimizer_method = config.get("optimizer_method", "eigenvector_follow")
    line_search = config.get("line_search", "NoLineSearch")
    ts_search = resolve_ts_search(config)
    dmax = config.get("dmax", 0.1)
    coordinate_type = config.get("coordinate_type", "TRIC")
    gsm_type = config.get("gsm_type", "DE_GSM")
    num_nodes = int(config.get("num_nodes", 11))
    add_node_tol = config.get("add_node_tol", 0.1)
    conv_tol = float(config.get("conv_tol", 5e-4))
    ediff = float(config.get("ediff", 100.0))
    fmax = float(config.get("fmax", 100.0))
    ID = config.get("ID", 0)
    max_gsm_steps = int(config.get("max_gsm_steps", 100))
    max_opt_steps = config.get("max_opt_steps", 3)
    fixed_reactant = config.get("fixed_reactant", False)
    fixed_product = config.get("fixed_product", False)
    output_dir = config.get("output_dir")  # writes string_<ID>.xyz checkpoints when set
    log_level = config.get("log_level", "INFO")

    if optimizer_method not in OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer_method: {optimizer_method}. Supported methods are {list(OPTIMIZERS)}.")

    # read atoms
    if gsm_type == "DE_GSM":
        assert len(atoms_list) == 2, "DE_GSM requires exactly two geometries (initial and final)."
        atoms_reactant: Atoms
        atoms_product: Atoms
        atoms_reactant, atoms_product = atoms_list[0], atoms_list[1]
    elif gsm_type == "SE_GSM":
        assert len(atoms_list) == 1, "SE_GSM requires exactly one geometry (initial)."
        atoms_reactant: Atoms = atoms_list[0]
        raise NotImplementedError("SE_GSM is not implemented yet.")
    else:
        raise ValueError(f"Unsupported gsm_type: {gsm_type}. Supported types are 'DE_GSM' and 'SE_GSM'.")

    # pygsm only logs; send its output to stdout
    output.setup_logging(log_level.upper())

    output.printcool("Preparing for GSM")
    # set level of theory
    calc = driver.to_ase_calc()
    lot = ASELoT(
        calculator=calc,
        geom=[[x.symbol, *x.position] for x in atoms_reactant],
        string_id=ID,
    )
    # build PES object
    pes_obj = PES(lot=lot, ad_idx=0)

    # build topology
    output.printcool("Building topologies")
    element_table = ElementData()
    elements = [element_table.from_symbol(sym) for sym in atoms_reactant.get_chemical_symbols()]
    topology_reactant = build_topology(
        xyz=atoms_reactant.get_positions(),
        atoms=elements,
    )
    topology_product = build_topology(
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
    connect = coordinate_type == "DLC"
    addtr = coordinate_type == "TRIC"
    addcart = coordinate_type == "HDLC"
    output.printcool("Building Primitive Internal Coordinates")
    prim_reactant = PrimitiveInternalCoordinates(
        xyz=atoms_reactant.get_positions(),
        atoms=elements,
        topology=topology_reactant,
        connect=connect,
        addtr=addtr,
        addcart=addcart,
    )
    prim_product = PrimitiveInternalCoordinates(
        xyz=atoms_product.get_positions(),
        atoms=elements,
        topology=topology_reactant,  # use reactant topology for product to ensure same set of primitives
        connect=connect,
        addtr=addtr,
        addcart=addcart,
    )
    # add product coords to reactant coords
    prim_reactant.add_union_primitives(prim_product)

    # delocalized internal coordinates
    output.printcool("Building Delocalized Internal Coordinates")
    deloc_coords_reactant = DelocalizedInternalCoordinates(
        xyz=atoms_reactant.get_positions(),
        atoms=elements,
        connect=connect,
        addtr=addtr,
        addcart=addcart,
        primitives=prim_reactant,
    )

    # molecules
    output.printcool(f"Building the reactant object with {coordinate_type}")
    form_hessian = optimizer_method == "eigenvector_follow"
    molecule_reactant = Molecule(
        geom=[[x.symbol, *x.position] for x in atoms_reactant],
        pes=pes_obj,
        coord_obj=deloc_coords_reactant,
        form_hessian=form_hessian,
    )
    molecule_product = Molecule.copy_from(
        molecule_reactant,
        xyz=atoms_product.get_positions(),
        new_node_id=num_nodes - 1,
    )

    # optimizer
    output.printcool("Building optimizer")
    optimizer_object = OPTIMIZERS[optimizer_method](
        print_level=1,
        linesearch=line_search,
        update_hess_in_bg=not (ts_search == "climb" or optimizer_method == "lbfgs"),
        conv_ediff=ediff,
        conv_gmax=fmax,
        dmax=dmax,
    )

    # GSM
    output.printcool("Building GSM object")
    gsm = DEGSM(
        reactant=molecule_reactant,
        product=molecule_product,
        nnodes=num_nodes,
        conv_tol=conv_tol,
        conv_gmax=fmax,
        conv_ediff=ediff,
        add_node_tol=add_node_tol,
        growth_direction=0,
        optimizer=optimizer_object,
        string_id=ID,
        print_level=1,
        mp_cores=1,
        interp_method="DLC",
        output_dir=output_dir,
    )

    # optimize reactant and product if not fixed
    if not fixed_reactant:
        output.printcool("Optimizing reactant")
        optimizer_object.optimize(
            molecule=molecule_reactant,
            ref_energy=molecule_reactant.energy,
            opt_steps=100,
        )
    if not fixed_product:
        output.printcool("Optimizing product")
        optimizer_object.optimize(
            molecule=molecule_product,
            ref_energy=molecule_product.energy,
            opt_steps=100,
        )

    # run GSM
    output.printcool("Running GSM")
    gsm.go_gsm(max_iters=max_gsm_steps, opt_steps=max_opt_steps, rtype=TS_SEARCH_RTYPE[ts_search])

    # write the results into an xyz file
    # add charge and multiplicity info to the frames
    string_ase, ts_ase = gsm_to_ase_atoms(gsm)
    charge = atoms_reactant.info.get("charge", 0)
    multiplicity = atoms_reactant.info.get("multiplicity", 1)
    for frame in string_ase:
        frame.info["charge"] = charge
        frame.info["multiplicity"] = multiplicity
    ts_ase.info["charge"] = charge
    ts_ase.info["multiplicity"] = multiplicity
    # write to xyz
    ase.io.write(f"{filename}_gsm.xyz", string_ase)
    ase.io.write(f"{filename}_ts.xyz", ts_ase)


def gsm_to_ase_atoms(gsm: Any):
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
