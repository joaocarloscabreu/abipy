import sys
import os
import abipy.data as abidata
import abipy.abilab as abilab
import abipy.flowtk as flowtk


def make_scf_input(usepaw=0, nspinor=1):
    """Returns input for GS-SCF calculation."""
    if nspinor == 1:
        pseudos = abidata.pseudos("14si.pspnc") if usepaw == 0 else abidata.pseudos("Si.GGA_PBE-JTH-paw.xml")
    else:
        pseudos = abidata.pseudos("Si_r.psp8") if usepaw == 0 else abidata.pseudos("Si.GGA_PBE-JTH-paw.xml")

    structure = dict(
         ntypat=1,
         natom=2,
         typat=[1, 1],
         znucl=14,
         #acell=3 * [10.26310667319252],
         acell=3 * [10.2073557], # 5.4015 Ang
         rprim=[[0.0,  0.5,  0.5],
                [0.5,  0.0,  0.5],
                [0.5,  0.5,  0.0]],
         xred=[[0.0 , 0.0 , 0.0],
               [0.25, 0.25, 0.25]],
    )

    scf_input = abilab.AbinitInput(structure=structure, pseudos=pseudos)

    # Global variables
    nband = 8 if nspinor == 1 else 16
    scf_input.set_vars(
        ecut=8,
        nband=nband,
        nspinor=nspinor,
        nstep=100,
        tolvrs=1e-8,
    )

    if scf_input.ispaw:
        scf_input.set_vars(pawecutdg=2 * scf_input["ecut"])

    # Set k-mesh
    scf_input.set_kmesh(ngkpt=[2, 2, 2], shiftk=[0, 0, 0])

    return scf_input


def build_flow(options):
    # Set working directory (default is the name of the script with '.py' removed and "run_" replaced by "flow_")
    if not options.workdir:
        options.workdir = os.path.basename(sys.argv[0]).replace(".py", "").replace("run_", "flow_")

    # Get the SCF input (without SOC)
    scf_input = make_scf_input(nspinor=1, usepaw=0)

    # Build the flow.
    from abipy.flowtk.eph_flows import EphARPESFlow

    kpath = [
            [ 0.0, 0.0, 0.0],
            [ 0.5, 0.0, 0.5],
            #[0.5000000000,    0.2500000000,    0.7500000000],
            #[0.3750000000,    0.3750000000,    0.7500000000],
            [0.0000000000,    0.0000000000,    0.0000000000],
            [0.5000000000,    0.5000000000,    0.5000000000],
            ]
    flow = EphARPESFlow.from_scf_input(options.workdir, scf_input, kmesh=[4,4,4], kbounds=kpath, bbounds=[2,8], with_stern = True)

    return flow


# This block generates the thumbnails in the Abipy gallery.
# You can safely REMOVE this part if you are using this script for production runs.
if os.getenv("READTHEDOCS", False):
    __name__ = None
    import tempfile
    options = flowtk.build_flow_main_parser().parse_args(["-w", tempfile.mkdtemp()])
    build_flow(options).graphviz_imshow()


@flowtk.flow_main
def main(options):
    """
    This is our main function that will be invoked by the script.
    flow_main is a decorator implementing the command line interface.
    Command line args are stored in `options`.
    """
    return build_flow(options)


if __name__ == "__main__":
    sys.exit(main())
