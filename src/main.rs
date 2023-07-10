mod gla_package;

use crate::gla_package::{gla::{
    aging_gompertz_makeham, fertility_brass_polynomial, find_maximum_fertility, gla_model,
    growth_function, learning_function,
}, simulate::run_simulation};

// use easybench::bench;

// use peroxide::fuga::{GaussLegendre, G7K15R};
// use peroxide::numerical::integral::{gauss_kronrod_quadrature, integrate};

fn main() {
    let time_step = 1.0;
    let initial_female_proportion = 0.5;
    let minimum_mortality = 1e-5;
    let aging_parameters = [1.15106610e-02, 2.71975789e-02, 4.26805906e-02];
    let learning_parameters = [2.93238557e-02, 3.94526937e+01, 9.41817943e-02];
    let growth_parameters: [f64; 2] = [1.00399978e-01, 9.23916941e-02];

    let female_fertility_parameters = [2.445e-5, 14.8, 32.836];
    let male_fertility_parameters = [2.445e-5, 14.8, 32.836];

    let female_fertility_function = fertility_brass_polynomial;
    let male_fertility_function = fertility_brass_polynomial;

    let female_maximum_fertility = find_maximum_fertility(
        &female_fertility_function,
        &female_fertility_parameters,
        20.0,
    );
    let male_maximum_fertility =
        find_maximum_fertility(&male_fertility_function, &male_fertility_parameters, 20.0);

    let normalized_male_fertility_closure = Box::new(|x: f64| -> f64 {
        (male_fertility_function(x, &male_fertility_parameters) / male_maximum_fertility).min(1.0)
    });

    let normalized_female_fertility_closure = Box::new(|x: f64| -> f64 {
        (female_fertility_function(x, &female_fertility_parameters) / female_maximum_fertility)
            .min(1.0)
    });

    let aging_intermediate_closure = |x: f64,
                                      aging_parameters: &[f64],
                                      learning_parameters: &[f64],
                                      growth_parameters: &[f64]|
     -> f64 {
        gla_model(
            x,
            aging_gompertz_makeham as fn(f64, &[f64]) -> f64,
            learning_function,
            growth_function,
            &aging_parameters,
            &learning_parameters,
            &growth_parameters,
            minimum_mortality,
        )
    };
    println!("Female maximum fertility : {}", female_maximum_fertility);
    println!("Male maximum fertility : {}", male_maximum_fertility);

    let initial_age_distribution = [20.0, 10.0];
    let initial_b_distribution = [0.07, 0.001];
    let initial_lmax_distribution = [0.0, 0.0];

    let population_cap = 10000;

    let mutable_b = true;
    let mutable_lmax = false;
    let b_mutation_rate: f64 = 0.02;
    let lmax_mutation_rate: f64 = 0.02;
    let b_mutation_strength = 0.006;
    let lmax_mutation_strength = 0.012;

    run_simulation(population_cap, &aging_parameters, &learning_parameters, &growth_parameters, initial_age_distribution, initial_b_distribution, initial_lmax_distribution, initial_female_proportion, time_step, mutable_b, mutable_lmax, b_mutation_rate, lmax_mutation_rate, b_mutation_strength, lmax_mutation_strength, aging_intermediate_closure, &normalized_male_fertility_closure, &normalized_female_fertility_closure)
}