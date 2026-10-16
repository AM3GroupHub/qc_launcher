"""Double-ended growing string method with pygsm (fork: github.com/X1X1010/pygsm)."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

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

from qc_launcher.drivers import BaseDriver
from qc_launcher.utils.config import ConfigDeprecationWarning

TS_SEARCH_RTYPE = {"none": 0, "climb": 1, "exact": 2}  # go_gsm rtype per ts_search setting
OPTIMIZERS = {"eigenvector_follow": EigenvectorFollow, "lbfgs": LBFGS}
COORDINATE_TYPES = ("DLC", "TRIC", "HDLC")


def resolve_ts_search(config: dict) -> str:
    """``ts_search`` (none | climb | exact), or the legacy ``climb`` flag.

    Legacy mapping, kept for existing configs: ``climb: true`` -> climbing image only,
    ``climb: false`` -> climbing image plus exact TS search.
    """
    ts_search = config.get("ts_search")
    if ts_search is None:
        if "climb" in config:
            warnings.warn(
                "[gsm] key 'climb' is deprecated, use 'ts_search: climb | exact | none'.", ConfigDeprecationWarning
            )
        return "climb" if config.get("climb", True) else "exact"
    if ts_search not in TS_SEARCH_RTYPE:
        raise ValueError(f"Unsupported ts_search: {ts_search}. Supported values are {list(TS_SEARCH_RTYPE)}.")
    return ts_search


def build_internal_coordinates(reactant: Atoms, product: Atoms, coordinate_type: str):
    """Delocalized internal coordinates of the reactant that also span the product's bonds."""
    if coordinate_type not in COORDINATE_TYPES:
        raise ValueError(f"Unsupported coordinate_type: {coordinate_type}. Use one of {COORDINATE_TYPES}.")
    element_table = ElementData()
    elements = [element_table.from_symbol(symbol) for symbol in reactant.get_chemical_symbols()]
    output.printcool("Building topologies")
    topology_reactant = build_topology(xyz=reactant.get_positions(), atoms=elements)
    topology_product = build_topology(xyz=product.get_positions(), atoms=elements)
    for bond in topology_product.edges():
        if not topology_reactant.has_edge(*bond):
            print(f"Adding bond {bond} to reactant topology")
            topology_reactant.add_edge(*bond)

    kwargs = {
        "atoms": elements,
        "connect": coordinate_type == "DLC",
        "addtr": coordinate_type == "TRIC",
        "addcart": coordinate_type == "HDLC",
    }
    output.printcool("Building Primitive Internal Coordinates")
    prim_reactant = PrimitiveInternalCoordinates(xyz=reactant.get_positions(), topology=topology_reactant, **kwargs)
    # the product uses the reactant topology so both share one set of primitives
    prim_product = PrimitiveInternalCoordinates(xyz=product.get_positions(), topology=topology_reactant, **kwargs)
    prim_reactant.add_union_primitives(prim_product)
    output.printcool("Building Delocalized Internal Coordinates")
    return DelocalizedInternalCoordinates(xyz=reactant.get_positions(), primitives=prim_reactant, **kwargs)


def write_results(gsm, reactant: Atoms, filename: str) -> None:
    """Write ``{filename}_gsm.xyz`` (string with energies) and ``{filename}_ts.xyz``."""
    info = {"charge": reactant.info.get("charge", 0), "multiplicity": reactant.info.get("multiplicity", 1)}
    frames = []
    for energy, geom in zip(gsm.energies, gsm.geometries):
        frame = Atoms(symbols=[x[0] for x in geom], positions=[x[1:4] for x in geom])
        frame.info.update(info, energy=energy)
        frames.append(frame)
    ts_geom = gsm.nodes[gsm.TSnode].geometry
    ts_atoms = Atoms(symbols=[x[0] for x in ts_geom], positions=[x[1:4] for x in ts_geom])
    ts_atoms.info.update(info)
    ase.io.write(f"{filename}_gsm.xyz", frames)
    ase.io.write(f"{filename}_ts.xyz", ts_atoms)


def run_gsm(driver: BaseDriver, config: dict, atoms_list: Sequence[Atoms], filename: str = "molecule") -> None:
    """Config keys follow the pygsm fork's parameter names (defaults): ``gsm_type`` (DE_GSM; SE_GSM is not
    implemented), ``nnodes`` (11), ``coordinate_type`` (TRIC | DLC | HDLC), ``optimizer_method``
    (eigenvector_follow | lbfgs), ``linesearch`` (NoLineSearch), ``ts_search`` (climb | exact | none), ``dmax``
    (0.1), ``add_node_tol`` (0.1), ``conv_tol`` (5e-4), ``conv_ediff`` / ``conv_gmax`` (100, i.e. disabled;
    pygsm's own defaults are 0.1 / 0.001), ``max_iters`` (100 GSM iterations), ``opt_steps`` (3 optimizer steps
    per GSM iteration), ``endpoint_opt_steps`` (100), ``fixed_reactant`` / ``fixed_product`` (False),
    ``string_id`` (0), ``output_dir``, ``log_level`` (INFO). Old spellings (num_nodes, max_gsm_steps,
    max_opt_steps, ID, fmax, ediff, line_search) are accepted with a deprecation warning.
    Requires exactly two input frames (reactant, product).
    """
    gsm_type = config.get("gsm_type", "DE_GSM")
    if gsm_type == "SE_GSM":
        raise NotImplementedError("SE_GSM is not implemented yet.")
    if gsm_type != "DE_GSM":
        raise ValueError(f"Unsupported gsm_type: {gsm_type}. Supported types are 'DE_GSM' and 'SE_GSM'.")
    if len(atoms_list) != 2:
        raise ValueError("DE_GSM requires exactly two geometries (reactant and product).")
    reactant, product = atoms_list

    optimizer_method = config.get("optimizer_method", "eigenvector_follow")
    if optimizer_method not in OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer_method: {optimizer_method}. Supported methods are {list(OPTIMIZERS)}.")
    linesearch = config.get("linesearch", "NoLineSearch")
    ts_search = resolve_ts_search(config)
    dmax = float(config.get("dmax", 0.1))
    coordinate_type = config.get("coordinate_type", "TRIC")
    nnodes = int(config.get("nnodes", 11))
    add_node_tol = float(config.get("add_node_tol", 0.1))
    conv_tol = float(config.get("conv_tol", 5e-4))
    conv_ediff = float(config.get("conv_ediff", 100.0))
    conv_gmax = float(config.get("conv_gmax", 100.0))
    string_id = int(config.get("string_id", 0))
    max_iters = int(config.get("max_iters", 100))
    opt_steps = int(config.get("opt_steps", 3))
    endpoint_opt_steps = int(config.get("endpoint_opt_steps", 100))
    fixed_reactant = bool(config.get("fixed_reactant", False))
    fixed_product = bool(config.get("fixed_product", False))
    output_dir = config.get("output_dir")  # writes string_<ID>.xyz checkpoints when set
    log_level = str(config.get("log_level", "INFO"))

    output.setup_logging(log_level.upper())  # pygsm only logs; send its output to stdout
    output.printcool("Preparing for GSM")
    lot = ASELoT(calculator=driver.to_ase_calc(), geom=[[x.symbol, *x.position] for x in reactant], string_id=string_id)
    pes_obj = PES(lot=lot, ad_idx=0)
    deloc_coords = build_internal_coordinates(reactant, product, coordinate_type)

    output.printcool(f"Building the reactant object with {coordinate_type}")
    molecule_reactant = Molecule(
        geom=[[x.symbol, *x.position] for x in reactant],
        pes=pes_obj,
        coord_obj=deloc_coords,
        form_hessian=optimizer_method == "eigenvector_follow",
    )
    molecule_product = Molecule.copy_from(molecule_reactant, xyz=product.get_positions(), new_node_id=nnodes - 1)

    output.printcool("Building optimizer")
    optimizer = OPTIMIZERS[optimizer_method](
        print_level=1,
        linesearch=linesearch,
        update_hess_in_bg=not (ts_search == "climb" or optimizer_method == "lbfgs"),
        conv_ediff=conv_ediff,
        conv_gmax=conv_gmax,
        dmax=dmax,
    )
    output.printcool("Building GSM object")
    gsm = DEGSM(
        reactant=molecule_reactant,
        product=molecule_product,
        nnodes=nnodes,
        conv_tol=conv_tol,
        conv_gmax=conv_gmax,
        conv_ediff=conv_ediff,
        add_node_tol=add_node_tol,
        growth_direction=0,
        optimizer=optimizer,
        string_id=string_id,
        print_level=1,
        mp_cores=1,
        interp_method="DLC",
        output_dir=output_dir,
    )

    for label, molecule, fixed in (
        ("reactant", molecule_reactant, fixed_reactant),
        ("product", molecule_product, fixed_product),
    ):
        if not fixed:
            output.printcool(f"Optimizing {label}")
            optimizer.optimize(molecule=molecule, ref_energy=molecule.energy, opt_steps=endpoint_opt_steps)

    output.printcool("Running GSM")
    gsm.go_gsm(max_iters=max_iters, opt_steps=opt_steps, rtype=TS_SEARCH_RTYPE[ts_search])
    write_results(gsm, reactant, filename)
