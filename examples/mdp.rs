use std::collections::HashSet;

use rand::{thread_rng, Rng};
use ttrs::{
    TTVec,
    TTf64,
};

#[derive(Clone, Debug)]
struct Rules(Vec<Vec<Vec<(usize, f64)>>>);

impl Rules {
    fn new_lattice(
        size: usize,
        obstacles: HashSet<(usize, usize)>,
        diffusion_prob: f64,
    ) -> Self {
        let target_prob = 1. - 4. * diffusion_prob;
        let mut graph: Vec<Vec<Vec<(usize, f64)>>> = Vec::with_capacity(size * size);
        graph.resize(size * size, vec![vec![]; 4]);
        for i in 0..size {
            for j in 0..size {
                if obstacles.get(&(i, j)).is_none() {
                    let i_next = if i < (size - 1) { i + 1 } else { 0 };
                    let i_prev = if i != 0 { i - 1 } else { size - 1 };
                    let j_next = if j < (size - 1) { j + 1 } else { 0 };
                    let j_prev = if j != 0 { j - 1 } else { size - 1 };
                    let transitions_per_action = unsafe { &mut graph.get_unchecked_mut(i + j * size)[..] };
                    if obstacles.get(&(i_next, j)).is_none() {
                        transitions_per_action[0].push((i_next + size * j, (target_prob + diffusion_prob)));
                        transitions_per_action[1].push((i_next + size * j, diffusion_prob));
                        transitions_per_action[2].push((i_next + size * j, diffusion_prob));
                        transitions_per_action[3].push((i_next + size * j, diffusion_prob));
                    }
                    if obstacles.get(&(i, j_next)).is_none() {
                        transitions_per_action[0].push((i + size * j_next, diffusion_prob));
                        transitions_per_action[1].push((i + size * j_next, (target_prob + diffusion_prob)));
                        transitions_per_action[2].push((i + size * j_next, diffusion_prob));
                        transitions_per_action[3].push((i + size * j_next, diffusion_prob));
                    }
                    if obstacles.get(&(i_prev, j)).is_none() {
                        transitions_per_action[0].push((i_prev + size * j, diffusion_prob));
                        transitions_per_action[1].push((i_prev + size * j, diffusion_prob));
                        transitions_per_action[2].push((i_prev + size * j, (target_prob + diffusion_prob)));
                        transitions_per_action[3].push((i_prev + size * j, diffusion_prob));
                    }
                    if obstacles.get(&(i, j_prev)).is_none() {
                        transitions_per_action[0].push((i + size * j_prev, diffusion_prob));
                        transitions_per_action[1].push((i + size * j_prev, diffusion_prob));
                        transitions_per_action[2].push((i + size * j_prev, diffusion_prob));
                        transitions_per_action[3].push((i + size * j_prev, (target_prob + diffusion_prob)));
                    }
                    transitions_per_action.iter_mut().for_each(|x| {
                        let norm = x.into_iter().map(|y| { y.1 }).sum::<f64>();
                        x.into_iter().for_each(|y| {
                            y.1 /= norm;
                        })
                    });
                }
            }
        }
        Self(graph)
    }

    unsafe fn update_state(&self, prev_state: Vec<f64>, action: usize) -> Vec<f64> {
        let mut new_state: Vec<f64> = vec![0.; prev_state.len()];
        for (src, transitions) in self.0.iter().enumerate() {
            let src_mass = *prev_state.get_unchecked(src);
            for (dst, prob) in transitions.get_unchecked(action) {
                *new_state.get_unchecked_mut(*dst) += src_mass * *prob;
            }
        }
        new_state
    }
}

fn get_reward(state: &[f64], size: usize, node: (usize, usize)) -> f64 {
    state[node.0 + node.1 * size]
}

// 0123456
// |||||||
//@@@@@@@@@
//@S# # # @-0
//@       @-1
//@ # # # @-2
//@       @-3
//@ # # # @-4
//@       @-5
//@ # # #T@-6
//@@@@@@@@@
unsafe fn run_mdp(seq: &[usize]) -> f64 {
    let rules = Rules::new_lattice(7, From::from([
        (0, 1), (0, 3), (0, 5),
        (2, 1), (2, 3), (2, 5),
        (4, 1), (4, 3), (4, 5),
        (6, 1), (6, 3), (6, 5),
    ]), 0.01);
    let mut state = vec![0.; 49];
    state[0] = 1.;
    for a in seq {
        state = unsafe { rules.update_state(state, *a) };
    }
    get_reward(&state, 7, (6, 6))
}


fn main() {
    let mut rng = thread_rng();
    let mode_dims = [4; 50];
    let sweeps_num = 4;
    let mut tt = TTVec::<f64>::ttcross(&mode_dims, 50, 0.0001, |x| unsafe { run_mdp(x) }, sweeps_num).unwrap();
    println!("Exact reward value vs predicted:");
    for _ in 0..10 {
        let random_seq: Vec<_> = (0..50).map(|_| { rng.gen::<usize>() % 4 }).collect();
        println!("{:?} vs {:?}", unsafe { run_mdp(&random_seq) }, tt.log_eval_index(&random_seq).unwrap().exp().re);
    }
    tt.set_into_left_canonical().unwrap();
    tt.truncate_left_canonical(1e-5).unwrap();
    println!("Bond dimensions:");
    println!("{:?}", tt.get_bonds());
    println!("vs number of possible states: 37.");
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};
    use crate::Rules;
    #[test]
    fn test_mdp() {
        let mut rng = thread_rng();
        let rules = Rules::new_lattice(6, From::from([(0, 1), (1, 1), (3, 5), (2, 1), (4, 2)]), 0.01);
        let mut state = vec![0.; 36];
        state[15] = 1.;
        for _ in 0..100 {
            let action = rng.gen::<usize>() % 4;
            state = unsafe { rules.update_state(state, action) };
        }
        for p in &state {
            assert!(*p >= 0.);
        }
        assert!((state.into_iter().sum::<f64>() - 1.).abs() < 1e-5)
    }
}