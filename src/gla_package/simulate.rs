use std::fs::File;
use csv::Writer;
use indicatif::{ProgressBar, ProgressStyle};

use crate::gla_package::agent_based::{
    get_death_population, get_population_b_stats, get_population_lmax_stats, get_reproduction_population,
    increment_age_population, initialize_population,
};

#[derive(serde::Serialize)]
struct SimulationResult {
    mean_b: f64,
    mean_lmax: f64,
    time: f64,
    replicate_id: i32,
}

pub fn run_simulation(
    output_writer: &mut Writer<File>,
    population_cap: usize,
    simulation_time:usize,
    replicate_id: i32,
    assortative_mating: bool,
    aging_parameters: &[f64],
    learning_parameters: &[f64],
    growth_parameters: &[f64],
    initial_age_distribution: [f64; 2],
    initial_b_distribution: [f64; 2],
    initial_lmax_distribution: [f64; 2],
    initial_female_proportion: f64,
    time_step: f64,
    mutable_b: bool,
    mutable_lmax: bool,
    b_mutation_rate: f64,
    lmax_mutation_rate: f64,
    b_mutation_strength: f64,
    lmax_mutation_strength: f64,
    aging_intermediate_closure: impl Fn(f64, &[f64], &[f64], &[f64]) -> f64 + Send + Sync,
    normalized_male_fertility_closure: &Box<impl Fn(f64) -> f64>, 
    normalized_female_fertility_closure: &Box<impl Fn(f64) -> f64>
) {
    // let mut wtr = Writer::from_path("foo.csv").unwrap();
    let mut population = initialize_population(
        population_cap,
        aging_parameters,
        learning_parameters,
        growth_parameters,
        initial_age_distribution,
        initial_b_distribution,
        initial_lmax_distribution,
        initial_female_proportion,
    );
    let bar = ProgressBar::new(simulation_time as u64);
    bar.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:50.cyan/blue} {pos:>7}/{len:7} {msg}",
        )
        .unwrap()
        .progress_chars("##-"),
    );
    for i in 0..simulation_time {
        get_death_population(&mut population, time_step, &aging_intermediate_closure);
        get_reproduction_population(
            &mut population,
            assortative_mating,
            &normalized_male_fertility_closure,
            &normalized_female_fertility_closure,
            population_cap,
            mutable_b,
            mutable_lmax,
            b_mutation_rate,
            lmax_mutation_rate,
            b_mutation_strength,
            lmax_mutation_strength,
        );
        increment_age_population(&mut population, time_step);
        let b_stats = get_population_b_stats(&population);
        let lmax_stats = get_population_lmax_stats(&population);

        let res = SimulationResult {
            mean_b: b_stats.0,
            mean_lmax: lmax_stats.0,
            time: (i as f64) * time_step,
            replicate_id: replicate_id,
        };
        let _ = output_writer.serialize(res);
        bar.inc(1);
    }
    bar.finish();
}
