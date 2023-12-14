use peroxide::fuga::GaussLegendre;
use peroxide::numerical::integral::integrate;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, Normal, num_traits::Float};
use rayon::prelude::*;
use std::iter::zip;

#[derive(Clone)]
pub struct Agent {
    pub age: f64,
    pub female: bool,
    pub aging_parameters: Vec<f64>,
    pub learning_parameters: Vec<f64>,
    pub growth_parameters: Vec<f64>,
}

pub fn initialize_population<'a, 'b>(
    initial_population_size: usize,
    aging_parameters: &[f64],
    learning_parameters: &[f64],
    growth_parameters: &[f64],
    initial_age_distribution: [f64; 2],
    initial_b_distribution: [f64; 2],
    initial_lmax_distribution: [f64; 2],
    initial_female_proportion: f64,
) -> Vec<Agent> {
    let mut population = Vec::with_capacity(initial_population_size);

    let age_dist = Normal::new(initial_age_distribution[0], initial_age_distribution[1]).unwrap();
    let b_dist = Normal::new(initial_b_distribution[0], initial_b_distribution[1]).unwrap();
    let lmax_dist =
        Normal::new(initial_lmax_distribution[0], initial_lmax_distribution[1]).unwrap();

    for _ in 0..initial_population_size {
        let age: f64 = age_dist.sample(&mut rand::thread_rng()).max(0.0).round();
        let female: bool = rand::random::<f64>() < initial_female_proportion;
        let b = b_dist.sample(&mut rand::thread_rng()).max(0.0);
        let lmax = lmax_dist.sample(&mut rand::thread_rng()).max(0.0);

        let mut agent_aging_parameters = aging_parameters.to_owned();
        agent_aging_parameters[1] = b;

        let mut agent_learning_parameters = learning_parameters.to_owned();
        agent_learning_parameters[0] = lmax;

        let agent_growth_parameters = growth_parameters.to_owned();

        let agent = Agent {
            age,
            female,
            aging_parameters: agent_aging_parameters,
            learning_parameters: agent_learning_parameters,
            growth_parameters: agent_growth_parameters,
        };

        population.push(agent);
    }
    population
}

pub fn get_proba_of_death_agent(
    agent: &Agent,
    time_step: f64,
    aging_intermediate_closure: &dyn Fn(f64, &[f64], &[f64], &[f64]) -> f64,
) -> f64 {
    integrate(
        |x: f64| -> f64 {
            aging_intermediate_closure(
                x,
                &agent.aging_parameters,
                &agent.learning_parameters,
                &agent.growth_parameters,
            )
        },
        (agent.age, agent.age + time_step),
        GaussLegendre(5),
    )
}

pub fn get_death_agent(
    agent: &Agent,
    time_step: f64,
    aging_intermediate_closure: &dyn Fn(f64, &[f64], &[f64], &[f64]) -> f64,
    remove_non_reproducing: bool,
    male_menopause: f64,
    female_menopause: f64,
) -> bool {
    let proba_of_death = get_proba_of_death_agent(agent, time_step, aging_intermediate_closure);
    if !remove_non_reproducing {
        return rand::random::<f64>() < proba_of_death;
    }

    let menopause_age = if agent.female { female_menopause } else { male_menopause };
    if menopause_age.is_nan() || agent.age <= menopause_age {
        return rand::random::<f64>() < proba_of_death;
    }
    true
}

pub fn get_death_population<F: Fn(f64, &[f64], &[f64], &[f64]) -> f64 + Send + Sync>(
    population: &mut Vec<Agent>,
    time_step: f64,
    aging_intermediate_closure: &F,
    remove_non_reproducing: bool,
    male_menopause: f64,
    female_menopause: f64,
) {
    let death_test_parallel = population
        .par_iter()
        .map(|agent| get_death_agent(agent, time_step, aging_intermediate_closure, remove_non_reproducing, male_menopause, female_menopause))
        .collect::<Vec<_>>();
    let mut dead_agent_indexes: Vec<usize> = death_test_parallel
        .iter()
        .enumerate()
        .filter(|&(_, &value)| value)
        .map(|(index, _)| index)
        .collect();

    dead_agent_indexes.sort();
    dead_agent_indexes.reverse();

    for index in dead_agent_indexes.iter() {
        population.swap_remove(*index);
    }
}

pub fn increment_age_population(population: &mut Vec<Agent>, time_step: f64) {
    for agent in population.iter_mut() {
        agent.age += time_step;
    }
}

pub fn sort_population_by_age(population: &mut Vec<Agent>) {
    population.sort_by(|a, b| a.age.partial_cmp(&b.age).unwrap());
}

pub fn create_couples(population: &Vec<Agent>) -> Vec<(&Agent, &Agent)> {
    let mut female_population = population
        .iter()
        .filter(|a| a.female)
        .map(|a| a)
        .collect::<Vec<_>>();
    let mut male_population = population
        .iter()
        .filter(|a| !a.female)
        .map(|a| a)
        .collect::<Vec<_>>();

    if male_population.len() > female_population.len() {
        male_population = male_population[..female_population.len()].to_vec();
    } else {
        female_population = female_population[..male_population.len()].to_vec();
    }

    zip(male_population, female_population).collect::<Vec<_>>()
}

pub fn reproduction_test_couple(
    couple: &(&Agent, &Agent),
    normalized_male_fertility_closure: &Box<impl Fn(f64) -> f64>,
    normalized_female_fertility_closure: &Box<impl Fn(f64) -> f64>,
) -> bool {
    let male_chance_to_reproduce = normalized_male_fertility_closure(couple.0.age);
    let female_chance_to_reproduce = normalized_female_fertility_closure(couple.1.age);

    (rand::random::<f64>() < male_chance_to_reproduce)
        && (rand::random::<f64>() < female_chance_to_reproduce)
}

pub fn mutate_parameter(param: &mut f64, mutation_rate: f64, mutation_strength: f64) {
    if rand::random::<f64>() < mutation_rate {
        let mutation_dist = Normal::new(*param, mutation_strength).unwrap();
        *param = mutation_dist.sample(&mut rand::thread_rng()).max(0.0);
    }
}

pub fn reproduction_couple(
    couple: &(&Agent, &Agent),
    aging_parameters: &[f64],
    learning_parameters: &[f64],
    growth_parameters: &[f64],
    mutable_b: bool,
    mutable_lmax: bool,
    b_mutation_rate: f64,
    lmax_mutation_rate: f64,
    b_mutation_strength: f64,
    lmax_mutation_strength: f64,
) -> Agent {
    let mut b = (couple.0.aging_parameters[1] + couple.1.aging_parameters[1]) / 2.0;
    if mutable_b {
        mutate_parameter(&mut b, b_mutation_rate, b_mutation_strength);
    }
    let mut lmax = (couple.0.learning_parameters[0] + couple.1.learning_parameters[0]) / 2.0;
    if mutable_lmax {
        mutate_parameter(&mut lmax, lmax_mutation_rate, lmax_mutation_strength);
    }

    let female: bool = rand::random::<f64>() < 0.5;

    let mut agent_aging_parameters = aging_parameters.to_owned();
    agent_aging_parameters[1] = b;

    let mut agent_learning_parameters = learning_parameters.to_owned();
    agent_learning_parameters[0] = lmax;

    let agent_growth_parameters = growth_parameters.to_owned();

    Agent {
        age: 0.0,
        female: female,
        aging_parameters: agent_aging_parameters,
        learning_parameters: agent_learning_parameters,
        growth_parameters: agent_growth_parameters,
    }
}

pub fn get_reproduction_population(
    population: &mut Vec<Agent>,
    assortative_mating: bool,
    normalized_male_fertility_closure: &Box<impl Fn(f64) -> f64>,
    normalized_female_fertility_closure: &Box<impl Fn(f64) -> f64>,
    population_cap: usize,
    mutable_b: bool,
    mutable_lmax: bool,
    b_mutation_rate: f64,
    lmax_mutation_rate: f64,
    b_mutation_strength: f64,
    lmax_mutation_strength: f64,
) {
    if assortative_mating {
        sort_population_by_age(population);
    }else{
        population.shuffle(&mut rand::thread_rng());
    }
    let couples = create_couples(population);
    let reproduction_test = couples
        .iter()
        .map(|couple| {
            reproduction_test_couple(
                couple,
                normalized_male_fertility_closure,
                normalized_female_fertility_closure,
            )
        })
        .collect::<Vec<_>>();

    let successful_couples_indexes: Vec<usize> = reproduction_test
        .iter()
        .enumerate()
        .filter(|&(_, &value)| value)
        .map(|(index, _)| index)
        .collect();

    let mut new_babies: Vec<Agent> = successful_couples_indexes
        .iter()
        .map(|index| {
            reproduction_couple(
                &couples[*index],
                &population[*index].aging_parameters,
                &population[*index].learning_parameters,
                &population[*index].growth_parameters,
                mutable_b,
                mutable_lmax,
                b_mutation_rate,
                lmax_mutation_rate,
                b_mutation_strength,
                lmax_mutation_strength,
            )
        })
        .collect();

    new_babies.shuffle(&mut rand::thread_rng());
    if new_babies.len() > population_cap as usize - population.len() {
        new_babies = new_babies[..(population_cap as usize - population.len())].to_vec();
    }

    population.extend(new_babies);
}

pub fn get_population_b_stats(population: &Vec<Agent>) -> (f64, f64) {
    let b_values = population
        .iter()
        .map(|agent| agent.aging_parameters[1])
        .collect::<Vec<_>>();

    let b_mean = b_values.par_iter().sum::<f64>() / b_values.len() as f64;

    let b_variance = b_values
        .par_iter()
        .map(|b| (b - b_mean).powi(2))
        .sum::<f64>()
        / b_values.len() as f64;

    (b_mean, b_variance)
}

pub fn get_population_lmax_stats(population: &Vec<Agent>) -> (f64, f64){
    let lmax_values = population
        .iter()
        .map(|agent| agent.learning_parameters[0])
        .collect::<Vec<_>>();

    let lmax_mean = lmax_values.par_iter().sum::<f64>() / lmax_values.len() as f64;

    let lmax_variance = lmax_values
        .par_iter()
        .map(|lmax| (lmax - lmax_mean).powi(2))
        .sum::<f64>()
        / lmax_values.len() as f64;

    (lmax_mean, lmax_variance)
}