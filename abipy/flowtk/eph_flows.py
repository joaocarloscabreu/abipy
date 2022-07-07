# coding: utf-8
"""
Flows for electron-phonon calculations (high-level interface)
"""
import numpy as np

from abipy.core.kpoints import kpath_from_bounds_and_ndivsm
from .works import Work, PhononWork, PhononWfkqWork
from .flows import Flow
from .nodes import FileNode 



class EphPotFlow(Flow):
    r"""
    This flow computes the e-ph scattering potentials on a q-mesh defined by ngqpt
    and a list of q-points (usually a q-path) specified by the user.
    The DFPT potentials on the q-mesh are merged in the DVDB located in the outdata
    of the second work while the DFPT potentials on the q-path are merged in the DVDB
    located in the outdata of the third work.
    These DVDB files are then passed to the EPH code to compute the average over the unit
    cell of the periodic part of the scattering potentials as a function of q.
    Results are stored in the V1QAVG.nc files of the outdata of the tasks in the fourth work.
    """

    @classmethod
    def from_scf_input(cls, workdir, scf_input, ngqpt, qbounds,
                       ndivsm=5, with_becs=True, ddk_tolerance=None, prepgkk=0, manager=None):
        """
        Build the flow from an input file representing a GS calculation.

        Args:
            workdir: Working directory.
            scf_input: Input for the GS SCF run.
            ngqpt: 3 integers defining the q-mesh.
            qbounds: List of boundaries defining the q-path used for the computation of the GKQ files.
                The q-path is automatically generated using `ndivsm` and the reciprocal-space metric.
                If `ndivsm` is 0, the code assumes that `qbounds` contains the full list of q-points
                and no pre-processing is performed.
            ndivsm: Number of points in the smallest segment of the path defined by `qbounds`.
                Use 0 to pass list of q-points.
            with_becs: Activate calculation of Electric field and Born effective charges.
            ddk_tolerance: dict {"varname": value} with the tolerance used in the DDK run if `with_becs`.
            prepgkk: 1 to activate computation of all 3 * natom perts (debugging option).
            manager: |TaskManager| object.
        """
        flow = cls(workdir=workdir, manager=manager)

        # First work with GS run.
        scf_task = flow.register_scf_task(scf_input)[0]

        # Second work to compute phonons on the input nqgpt q-mesh.
        work_qmesh = PhononWork.from_scf_task(scf_task, qpoints=ngqpt, is_ngqpt=True,
                                              with_becs=with_becs, ddk_tolerance=ddk_tolerance)
        flow.register_work(work_qmesh)

        if ndivsm > 0:
            # Generate list of q-points from qbounds and ndivsm.
            qpath_list = kpath_from_bounds_and_ndivsm(qbounds, ndivsm, scf_input.structure)
        elif ndivsm == 0:
            # Use input list of q-points.
            qpath_list = np.reshape(qbounds, (-1, 3))
        else:
            raise ValueError("ndivsm cannot be negative. Received ndivsm: %s" % ndivsm)

        # Third Work: compute WFK/WFQ and phonons for qpt in qpath_list.
        # Don't include BECS because they have been already computed in the previous work.
        work_qpath = PhononWfkqWork.from_scf_task(
                       scf_task, qpath_list, ph_tolerance=None, tolwfr=1.0e-22, nband=None,
                       with_becs=False, ddk_tolerance=None, shiftq=(0, 0, 0), is_ngqpt=False, remove_wfkq=True,
                       prepgkk=prepgkk, manager=manager)

        flow.register_work(work_qpath)

        # Now we compute matrix elements fully ab-initio for each q-point.
        eph_work = Work()

        for eph_task in (-15, 15):
            eph_inp = scf_input.new_with_vars(
                optdriver=7,
                ddb_ngqpt=ngqpt,    # q-mesh associated to the DDB file.
                #dvdb_ngqpt=ngqpt,  # q-mesh associated to the DDVDB file.
                prtphdos=0,
                eph_task=eph_task
            )

            if eph_task == -15:
                # Use DVDB with ab-initio POTS along q-path to produce V1QAVG
                deps = {work_qmesh: "DDB", work_qpath: "DVDB"}
            elif eph_task == 15:
                # Use q-mesh to interpolate along the same q-path as above.
                deps = {work_qmesh: ["DDB", "DVDB"]}
                eph_inp.set_vars(ph_nqpath=len(qpath_list), ph_qpath=qpath_list)

            eph_work.register_eph_task(eph_inp, deps=deps)

        flow.register_work(eph_work)

        return flow


class GkqPathFlow(Flow):
    r"""
    This flow computes the gkq e-ph matrix elements <k+q|\Delta V_q|k> for a list of q-points (usually a q-path).
    The results are stored in the GKQ.nc files for the different q-points. These files can be used to analyze the behaviour
    of the e-ph matrix elements as a function of qpts with the the objects provided by the abipy.eph.gkq module.
    It is also possible to compute the e-ph matrix elements using the interpolated DFPT potentials
    if test_ft_interpolation is set to True.
    """

    @classmethod
    def from_scf_input(cls, workdir, scf_input, ngqpt, qbounds,
                       ndivsm=5, with_becs=True, ddk_tolerance=None,
                       test_ft_interpolation=False, prepgkk=0, manager=None):
        """
        Build the flow from an input file representing a GS calculation.

        Args:
            workdir: Working directory.
            scf_input: Input for the GS SCF run.
            ngqpt: 3 integers defining the q-mesh.
            qbounds: List of boundaries defining the q-path used for the computation of the GKQ files.
                The q-path is automatically generated using `ndivsm` and the reciprocal-space metric.
                If `ndivsm` is 0, the code assumes that `qbounds` contains the full list of q-points
                and no pre-processing is performed.
            ndivsm: Number of points in the smallest segment of the path defined by `qbounds`.
                Use 0 to pass list of q-points.
            with_becs: Activate calculation of Electric field and Born effective charges.
            ddk_tolerance: dict {"varname": value} with the tolerance used in the DDK run if `with_becs`.
            test_ft_interpolation: True to add an extra Work in which the GKQ files are computed
                using the interpolated DFPT potentials and the q-mesh defined by `ngqpt`.
                The quality of the interpolation depends on the convergence of the BECS, epsinf and `ngqpt`.
            prepgkk: 1 to activate computation of all 3 * natom perts (debugging option).
            manager: |TaskManager| object.
        """
        flow = cls(workdir=workdir, manager=manager)

        # First work with GS run.
        scf_task = flow.register_scf_task(scf_input)[0]

        # Second work to compute phonons on the input nqgpt q-mesh.
        work_qmesh = PhononWork.from_scf_task(scf_task, qpoints=ngqpt, is_ngqpt=True,
                                              with_becs=with_becs, ddk_tolerance=ddk_tolerance)
        flow.register_work(work_qmesh)

        if ndivsm > 0:
            # Generate list of q-points from qbounds and ndivsm.
            qpath_list = kpath_from_bounds_and_ndivsm(qbounds, ndivsm, scf_input.structure)
        elif ndivsm == 0:
            # Use input list of q-points.
            qpath_list = np.reshape(qbounds, (-1, 3))
        else:
            raise ValueError("ndivsm cannot be negative. Received ndivsm: %s" % ndivsm)

        # Third Work. Compute WFK/WFQ and phonons for qpt in qpath_list.
        # Don't include BECS because they have been already computed in the previous work.
        work_qpath = PhononWfkqWork.from_scf_task(
                       scf_task, qpath_list, ph_tolerance=None, tolwfr=1.0e-22, nband=None,
                       with_becs=False, ddk_tolerance=None, shiftq=(0, 0, 0), is_ngqpt=False, remove_wfkq=False,
                       prepgkk=prepgkk, manager=manager)

        flow.register_work(work_qpath)

        def make_eph_input(scf_inp, ngqpt, qpt):
            """
            Build input file to compute GKQ.nc file from GS SCF input.
            The calculation requires GS wavefunctions WFK, WFQ, a DDB file and a DVDB file
            """
            return scf_inp.new_with_vars(
                optdriver=7,
                eph_task=-2,
                nqpt=1,
                qpt=qpt,
                ddb_ngqpt=ngqpt,  # q-mesh associated to the DDB file.
                prtphdos=0,
            )

        # Now we compute matrix elements fully ab-initio for each q-point.
        eph_work = Work()

        qseen = set()
        for task in work_qpath.phonon_tasks:
            qpt = tuple(task.input["qpt"])
            if qpt in qseen: continue
            qseen.add(qpt)
            t = eph_work.register_eph_task(make_eph_input(scf_input, ngqpt, qpt), deps=task.deps)
            t.add_deps({work_qmesh: "DDB", work_qpath: "DVDB"})

        flow.register_work(eph_work)

        # Here we build another work to compute the gkq matrix elements
        # with interpolated potentials along the q-path.
        # The potentials are interpolated using the input ngqpt q-mesh.
        if test_ft_interpolation:
            inteph_work = Work()
            qseen = set()
            for task in work_qpath.phonon_tasks:
                qpt = tuple(task.input["qpt"])
                if qpt in qseen: continue
                qseen.add(qpt)
                eph_inp = make_eph_input(scf_input, ngqpt, qpt)
                # Note eph_use_ftinterp 1 to force the interpolation of the DFPT potentials with eph_task -2.
                eph_inp["eph_use_ftinterp"] = 1
                t = inteph_work.register_eph_task(eph_inp, deps=task.deps)
                t.add_deps({work_qmesh: ["DDB", "DVDB"]})
            flow.register_work(inteph_work)

        return flow

class EphARPESWork(Work):
    r"""
    This work calculates the self-energy for all the k-point in a high-symmetry path where the boundaries points are given.

    Input arguments:
    scf_input = AbinitInput object
    kbounds = list of high-symmetry k-points path
    bbounds = list of bands for the calculation with two elements: [ initial band, final band ]
    nodes = list of Node objects containing three elements: [ wfk_node, ddb_node, dvdb_node ]
    paths = list of paths containing three elements: [ wfk_path, ddb_path, dvdb_path ]
    stern = string with the path to POT file for sternheimer calculation
    manager = Flow Manager

    One can chose one permutation that includes at least access to wfk, ddb and dvdb files.
    Ex: nodes = [ wfk_node, None, None ] ; paths = [ None, ddb_path, dvdb_path ]
    """

    @classmethod
    def from_scf_input(cls, scf_input, kbounds, bbounds, nodes = None, paths = None, stern = None,  manager=None):

        from abipy.core.kpoints import find_irred_kpoints_generic, Kpoint
        from abipy.dfpt.ddb import DdbFile
        from abipy.flowtk.nodes import FileNode
        import os
        new = cls(manager=manager)
        # Keep a copy of the initial input.
        new.scf_input = scf_input.deepcopy()

        
        # Set the wfk, ddb and dvdb options
        new.wfk_node = None
        new.ddb_node = None
        new.dvdb_node = None
        if paths is not None:
            if paths[0] is not None:
                new.wfk_node = FileNode(paths[0])
            if paths[1] is not None:
                new.ddb_node = FileNode(paths[1])
                with DdbFile(paths[1]) as ddb:
                    ngqpt = ddb.guessed_ngqpt
                new.ddb_ngqpt = ngqpt# DdbFile.as_ddb(paths[1])
            if paths[2] is not None:
                new.dvdb_node = FileNode(paths[2])
        if nodes is not None:
            if nodes[0] is not None:
                new.wfk_node = nodes[0]
            if nodes[1] is not None:
                new.ddb_node = nodes[1]
                with DdbFile(self.ddb_node.filepath) as ddb:
                    ngqpt = ddb.guessed_ngqpt
                new.ddb_ngqpt = ngqpt # DdbFile.as_ddb(os.path.abspath(nodes[1]))
            if nodes[2] is not None:
                new.dvdb_node = nodes[2]

        if new.wfk_node is None:
            raise ValueError("Missing WFK file")
        if new.ddb_node is None:
            raise ValueError("Missing DDB file")
        if new.dvdb_node is None:
            raise ValueError("Missing DVDB file")

        

        # Find the k-points in the IBZ inside the WFK file
        IBZ = np.array([])
        with new.wfk_node.open_gsr() as gsr:
            lattice = gsr.ebands.reciprocal_lattice
            symmetries = gsr.ebands.structure.abi_spacegroup
            kmesh = np.diag(gsr.ebands.kpoints.ksampling.kptrlatt)
            for ik in gsr.ebands.kpoints:
                IBZ = np.vstack([IBZ,ik.frac_coords]) if IBZ.size else np.array([ik.frac_coords])

        # Check to which high-symmetry points given by kbounds corresponds to the IBZ points on the WFK file
        kbounds_sym = np.array([])
        for ik in kbounds:
            kpoint = Kpoint(ik,lattice)
            flag = False
            for iksym in kpoint.compute_star(symmetries).frac_coords:
                if np.any(np.all(np.abs(iksym - IBZ) < 0.0001,axis=1)):
                    idx = np.where(np.all(np.abs(iksym - IBZ) < 0.0001,axis=1))[0][0]
                    kbounds_sym = np.vstack([kbounds_sym,IBZ[idx]]) if kbounds_sym.size else np.array([IBZ[idx]])
                    flag = True
            if not flag:
                raise ValueError("[" + ", ".join(str(e) for e in ik) + "]" + " is not inside WFK file")

        # Set up the path between the points in kbounds that are inside the IBZ
        list_kpoints = np.array([kbounds_sym[0]])
        for ik in range(1,len(kbounds_sym)):
            
            direction = kbounds_sym[ik] - kbounds_sym[ik-1]
            norm_both = np.linalg.norm(direction) * np.linalg.norm(IBZ[1:],axis=1)

            line_highsymkpt = np.dot(direction, np.transpose(IBZ[1:]))/norm_both
            line_idx = np.append( np.where(line_highsymkpt > 0.999)[0], np.flip(np.where(line_highsymkpt < -0.999)[0]) )
            if np.all(np.abs(IBZ[line_idx[0]+1] - list_kpoints[-1]) < 0.00001 ):
                line_idx = np.delete(line_idx,0)
            for ik_line in line_idx: 
                list_kpoints = np.vstack([list_kpoints,IBZ[ik_line+1]])
            if np.all(kbounds_sym[ik] - np.array([0,0,0]) < 0.001):
                list_kpoints = np.vstack([list_kpoints,kbounds_sym[ik]])

        # Prepare the eph input file
        eph_deps = {new.wfk_node: "WFK", new.ddb_node:"DDB", new.dvdb_node : "DVDB"}

        new.eph_input = new.scf_input.new_with_vars(
                ngkpt=kmesh,
                optdriver=7,
                ddb_ngqpt=new.ddb_ngqpt,
                eph_intmeth=1,
                eph_ngqpt_fine=kmesh,
                eph_task=4,
                )

        # If sternheimer equations are being used
        if stern is not None:
            if stern[0] != "'" or stern[0] != '"':
                stern = '"' + stern + '"'
            elif stern[0] == "'":
                stern = '"' + stern[1:-2] + '"'
            new.eph_input.set_vars(
                    eph_stern=1,
                    getpot_filepath=stern
                    )
        # Set the bands from bbounds list
        new.bbounds = np.array(bbounds)

        if new.bbounds.shape != list_kpoints.shape:
            new.bbounds = np.repeat([new.bbounds],list_kpoints.shape[0],axis=0)

        new.eph_input.set_kptgw(list_kpoints, new.bbounds)
        
        new.register_eph_task(new.eph_input, deps=eph_deps)
        return new


class EphARPESFlow(Flow):
    r"""
    This flow calculates the self-energy for all the k-point in a high-symmetry path where the boundaries points are given.

    Input arguments:
    workdir = string with path of the workdir
    scf_input = AbinitInput object
    kmesh = list with k-points defining the IBZ 
    kbounds = list of high-symmetry k-points path
    bbounds = list of bands for the calculation with two elements: [ initial band, final band ]
    stern = boolean to select sternheimer equations
    manager = Flow Manager

    """


    @classmethod
    def from_scf_input(cls, workdir, scf_input, kmesh, kbounds, bbounds, with_stern = False,  manager=None):

        new = cls(workdir=workdir, manager=manager)
        new.scf_input = scf_input.deepcopy()
        new.kbounds = kbounds
        new.bbounds = bbounds
        new.with_stern = with_stern
                
        # Register the ground state calculation
        new.gs_work = new.register_scf_task(new.scf_input)
        new.scf_task = new.gs_work[0]

        # Set the NSCF inout file and register the calculation
        new.nscf_input = new.scf_input.new_with_vars(
                    ngkpt=kmesh,
        )
        if with_stern:
            new.nscf_input.set_vars(
                    prtpot=1
                    )
        new.gs_work.register_nscf_task(new.nscf_input,deps = {new.scf_task: "DEN"})
        new.nscf_task = new.gs_work[1]

        # Prepare the phonon calculation
        ph_work = PhononWork.from_scf_input(new.scf_input, qpoints=new.scf_input["ngkpt"],
                    is_ngqpt=True, with_becs=True,
                    ddk_tolerance=None)
        new.ph_work = new.register_work(ph_work)

        
        return new


    def on_all_ok(self):

        if self.with_stern:
            stern = self.nscf_task.outdir.has_abiext("POT")
        else:
            stern = None

        # After electronic and phonon ground state calculations prepare the eph through the desired path
        work  = EphARPESWork.from_scf_input(self.scf_input, self.kbounds, self.bbounds,paths=[None, self.ph_work.out_ddb, self.ph_work.out_dvdb], nodes= [self.nscf_task, None, None], stern = stern)

        self.register_work(work)
        self.allocate()
        self.build_and_pickle_dump()
        self.finalized = False

        return super().on_all_ok()


