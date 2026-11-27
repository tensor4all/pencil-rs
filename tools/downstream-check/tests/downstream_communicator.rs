//! A downstream crate owns its own `rsmpi-rt` dependency, initializes MPI itself and
//! hands its communicator to pencil-array. This is only possible if both sides resolve
//! to the *same* `rsmpi-rt` crate; with two versions the `Communicator` types differ and
//! this file does not compile.

use mpi::traits::{Communicator as _, CommunicatorCollectives as _};
use pencil_array::{AxisPermutation, ExtraShape, MpiTopology, Pencil, PencilArray};

#[test]
fn downstream_communicator_is_accepted_by_pencil_array() {
    // MPI is initialized (and finalized on drop) once, by the downstream crate.
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size() as usize;
    let topology = MpiTopology::<1>::new(&world, [size]).unwrap();
    // pencil-array's communicator type is the downstream's own `mpi` type.
    let _: &mpi::topology::CartesianCommunicator = topology.communicator();
    assert_eq!(topology.size(), size);
    assert_eq!(topology.rank(), world.rank());

    // A distributed object built from that communicator, and a collective over it.
    let pencil = Pencil::new_permuted(
        topology.clone(),
        [size * 2 + 1, 3],
        [0],
        AxisPermutation::new([1, 0]).unwrap(),
    )
    .unwrap();
    let array =
        PencilArray::from_elem(pencil.clone(), ExtraShape::new([1]).unwrap(), 1i64).unwrap();
    let local: i64 = array.as_slice().iter().sum();
    let mut total = 0i64;
    world.all_reduce_into(&local, &mut total, mpi::collective::SystemOperation::sum());
    assert_eq!(total, ((size * 2 + 1) * 3) as i64);
}
