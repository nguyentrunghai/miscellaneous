
from sklearn.decomposition import PCA

def cal_pca(traj, n_components):
    """
    traj:    mdtraj.Trajectory
    n_components:   int,    number of pc to output
    return projections on pc
    """
    pca = PCA(n_components = n_components)
    traj.superpose(traj, 0)
    reduced_cartesian = pca.fit_transform(traj.xyz.reshape( traj.n_frames, traj.n_atoms*3 ))
    #print "shape of reduced_cartesian ", reduced_cartesian.shape
    return reduced_cartesian

def cal_pca_mult_trajs(trajs, n_components=2):
    """
    trajs:  list of mdtraj.Trajectory
    n_components:   int,    number of pc to output
    return a list of n reduced_cartesian
    """
    if len(trajs) > 1:
        combined_traj = trajs[0].join(trajs[1:], check_topology=True)
    else:
        raise RuntimeError("number of trajs less than 2")

    rc = cal_pca(combined_traj, n_components)
    reduced_cartesian = []
    start = 0
    for traj in trajs:
        reduced_cartesian.append( rc[ start : start+len(traj) ] )
        start += len(traj)

    for i, r in enumerate(reduced_cartesian):
        print "shape of traj ", i, r.shape
    return reduced_cartesian


def cal_pca_use_ref(ref_trajs, trajs, n_components):
    """
    ref_trajs   :   list of mdtraj trajectories, used to define pc coordinates
    trajs       :   list of mdtraj trajectories, whose coordinates will be projected
    """
    assert type(ref_trajs) == list, "ref_trajs must be a list"
    assert type(trajs)     == list, "trajs must be a list"

    if len(ref_trajs) > 1:
        conbined_ref = ref_trajs[0].join(ref_trajs[1:], check_topology=True)
    else:
        conbined_ref = ref_trajs[0]
    conbined_ref.superpose(conbined_ref, 0)

    pca = PCA(n_components = n_components)
    rc_ref = pca.fit_transform( conbined_ref.xyz.reshape(conbined_ref.n_frames, conbined_ref.n_atoms*3) )

    reduced_cartesian_refs = []
    start = 0
    for t in ref_trajs:
        reduced_cartesian_refs.append( rc_ref[ start : start+len(t) ] )
        start += len(t)

    reduced_cartesian_trajs = []
    for traj in trajs:
        traj.superpose(conbined_ref, 0)
        rc = pca.transform( traj.xyz.reshape(traj.n_frames, traj.n_atoms*3) )
        reduced_cartesian_trajs.append(rc)

    return reduced_cartesian_refs, reduced_cartesian_trajs


