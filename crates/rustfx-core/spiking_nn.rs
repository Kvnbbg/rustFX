// spiking_nn.rs - Simple Spiking Neural Network (LIF model)
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;

#[derive(Debug, Clone)]
pub struct LifNeuron {
    membrane_potential: f64,
    threshold: f64,
    reset_potential: f64,
    tau: f64, // Time constant
    refractory_period: f64,
    time_since_spike: f64,
    last_spike_time: f64,
    spiked: bool,
}

impl LifNeuron {
    pub fn new() -> Self {
        LifNeuron {
            membrane_potential: 0.0,
            threshold: 1.0,
            reset_potential: 0.0,
            tau: 20.0,
            refractory_period: 5.0,
            time_since_spike: 0.0,
            last_spike_time: -1.0,
            spiked: false,
        }
    }

    pub fn integrate(&mut self, input_current: f64, dt: f64, current_time: f64) {
        self.time_since_spike += dt;
        if self.time_since_spike < self.refractory_period {
            self.membrane_potential = self.reset_potential;
            self.spiked = false;
            return;
        }
        self.membrane_potential += dt * (-self.membrane_potential / self.tau + input_current);
        self.spiked = if self.membrane_potential >= self.threshold {
            self.membrane_potential = self.reset_potential;
            self.time_since_spike = 0.0;
            self.last_spike_time = current_time;
            true
        } else {
            false
        };
    }
}

#[derive(Debug, Clone)]
pub struct SpikingNet {
    neurons: Vec<LifNeuron>,
    weights: Vec<Vec<f64>>, // Connectivity
    a_plus: f64,
    a_minus: f64,
    tau_plus: f64,
    tau_minus: f64,
    learning_rate: f64,
    min_weight: f64,
    max_weight: f64,
    sim_time: f64,
}

impl SpikingNet {
    pub fn new(num_neurons: usize) -> Self {
        let mut rng = thread_rng();
        let uniform = Uniform::from(-0.1..0.1);
        let weights: Vec<Vec<f64>> = (0..num_neurons).map(|_| (0..num_neurons).map(|_| uniform.sample(&mut rng)).collect()).collect();
        let neurons = (0..num_neurons).map(|_| LifNeuron::new()).collect();
        SpikingNet {
            neurons,
            weights,
            a_plus: 0.01,
            a_minus: -0.012,
            tau_plus: 20.0,
            tau_minus: 20.0,
            learning_rate: 0.001,
            min_weight: -1.0,
            max_weight: 1.0,
            sim_time: 0.0,
        }
    }

    pub fn step(&mut self, inputs: &[f64], dt: f64) -> Vec<bool> {
        self.sim_time += dt;
        let mut currents = inputs.to_vec();
        for i in 0..self.neurons.len() {
            for j in 0..self.neurons.len() {
                if self.neurons[j].spiked {
                    currents[i] += self.weights[i][j];
                }
            }
        }
        let mut spikes = vec![false; self.neurons.len()];
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            neuron.integrate(currents[i], dt, self.sim_time);
            spikes[i] = neuron.spiked;
        }
        self.apply_stdp();
        spikes
    }

    fn apply_stdp(&mut self) {
        for post_idx in 0..self.neurons.len() {
            for pre_idx in 0..self.neurons.len() {
                let post_time = self.neurons[post_idx].last_spike_time;
                let pre_time = self.neurons[pre_idx].last_spike_time;
                if post_time < 0.0 || pre_time < 0.0 {
                    continue;
                }
                let delta_t = post_time - pre_time;
                let delta_w = if delta_t > 0.0 {
                    self.a_plus * (-delta_t / self.tau_plus).exp()
                } else {
                    self.a_minus * (delta_t / self.tau_minus).exp()
                };
                let updated = self.weights[post_idx][pre_idx] + self.learning_rate * delta_w;
                self.weights[post_idx][pre_idx] = updated.clamp(self.min_weight, self.max_weight);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoihiEmulator {
    net: SpikingNet,
    pending_spikes: Vec<(usize, f64)>,
}

impl LoihiEmulator {
    pub fn new(num_neurons: usize) -> Self {
        Self {
            net: SpikingNet::new(num_neurons),
            pending_spikes: Vec::new(),
        }
    }

    pub fn inject_spike(&mut self, neuron_idx: usize, time: f64) {
        self.pending_spikes.push((neuron_idx, time));
    }

    pub fn step_event(&mut self, dt: f64) -> Vec<bool> {
        let mut inputs = vec![0.0; self.net.neurons.len()];
        for (neuron_idx, _) in self.pending_spikes.drain(..) {
            if let Some(input) = inputs.get_mut(neuron_idx) {
                *input += 1.0;
            }
        }
        self.net.step(&inputs, dt)
    }

    pub fn network(&self) -> &SpikingNet {
        &self.net
    }
}
