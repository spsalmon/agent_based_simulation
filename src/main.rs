mod gla_package;
use csv::Writer;
use crate::gla_package::{gla::{
    aging_gompertz_makeham, fertility_brass_polynomial, find_maximum_fertility, gla_model,
    growth_function, learning_function, constant_fertility,
}, simulate::run_simulation};

// use easybench::bench;

// use peroxide::fuga::{GaussLegendre, G7K15R};
// use peroxide::numerical::integral::{gauss_kronrod_quadrature, integrate};

fn main() {
    let time_step = 1.0;
    let initial_female_proportion = 0.5;
    let minimum_mortality = 1e-5;
    let aging_parameters = [0.00275961297460256,0.04326224872667336,0.025201676835511704] ;
    let learning_parameters = [0.01606792505529796,39.006865144958745,0.11060749334680318];
    let growth_parameters: [f64; 2] = [0.05168141300917714,0.08765165352033985];

    // let female_fertility_parameters = [1.0];
    // let male_fertility_parameters = [1.0];

    // let female_fertility_function = constant_fertility;
    // let male_fertility_function = constant_fertility;

    let female_fertility_parameters = [2.445e-5, 14.8, 32.836];
    let male_fertility_parameters = [2.445e-5, 14.8, 32.836];

    let female_menopause = female_fertility_parameters[1] + female_fertility_parameters[2];
    let male_menopause = male_fertility_parameters[1] + male_fertility_parameters[2];

    // let male_fertility_parameters = [0.00000978, 14.8, 47.836];

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

    let normalized_female_fertility_closure = Box::new(|x: f64| -> f64{
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
    // println!("Female maximum fertility : {}", female_maximum_fertility);
    // println!("Male maximum fertility : {}", male_maximum_fertility);

    let initial_age_distribution = [20.0, 10.0];
    let initial_b_distribution = [0.14, 0.005];
    let mut initial_lmax_distribution = [0.15, 0.0];
    let mut initial_gmax_distribution = [0.0, 0.0];

    let population_cap = 10000;
    let simulation_time : usize = 600000;
    let replicate_number = 5;
    let assortative_mating = true;
    let remove_non_reproducing = true;

    let mutable_b = true;
    let mutable_lmax = false;
    let mutable_gmax = false;
    let b_mutation_rate: f64 = 0.02;
    let lmax_mutation_rate: f64 = 0.02;
    let gmax_mutation_rate: f64 = 0.02;
    let b_mutation_strength = 0.012;
    let lmax_mutation_strength = 0.012;
    let gmax_mutation_strength = 0.012;

    let base_name_part = "plateau_brass_polynomial_equal_both";
    // let base_name_part = "early_slope_brass_polynomial_equal_both";
    let mut learning_name_part = "with_learning";
    let mut mating_name_part = "random_mating";
    let mut removal_name_part = "non_reproducing_kept";

    if assortative_mating{
        mating_name_part = "assortative_mating";
    }

    if remove_non_reproducing{
        removal_name_part = "non_reproducing_removed";
    }

    println!("######################################");
    println!("###### Simulation with learning ######");
    println!("######################################");

    let output_file_name = format!("./simulation_results/{}_{}_{}_{}_{}.csv", base_name_part, mating_name_part, learning_name_part, removal_name_part,initial_lmax_distribution[0]);
    let mut wtr = Writer::from_path(output_file_name).unwrap();

    for i in 0..replicate_number{
        println!("Replicate : {}/{}", i+1, replicate_number);
        run_simulation(&mut wtr, population_cap, simulation_time, i, assortative_mating, &aging_parameters, &learning_parameters, &growth_parameters, initial_age_distribution, initial_b_distribution, initial_lmax_distribution, initial_female_proportion, time_step, mutable_b, mutable_lmax, b_mutation_rate, lmax_mutation_rate, b_mutation_strength, lmax_mutation_strength, aging_intermediate_closure, &normalized_male_fertility_closure, &normalized_female_fertility_closure, remove_non_reproducing, male_menopause, female_menopause)
    }

    println!("#########################################");
    println!("###### Simulation without learning ######");
    println!("#########################################");
    initial_lmax_distribution = [0.0, 0.0];
    learning_name_part = "no_learning";

    let output_file_name = format!("./simulation_results/{}_{}_{}_{}.csv", base_name_part, mating_name_part, learning_name_part, removal_name_part);
    let mut wtr = Writer::from_path(output_file_name).unwrap();

    for i in 0..replicate_number{
        println!("Replicate : {}/{}", i+1, replicate_number);
        run_simulation(&mut wtr, population_cap, simulation_time, i, assortative_mating, &aging_parameters, &learning_parameters, &growth_parameters, initial_age_distribution, initial_b_distribution, initial_lmax_distribution, initial_gmax_distribution, initial_female_proportion, time_step, mutable_b, mutable_lmax, mutable_gmax, b_mutation_rate, lmax_mutation_rate, gmax_mutation_rate, b_mutation_strength, lmax_mutation_strength, gmax_mutation_strength, aging_intermediate_closure, &normalized_male_fertility_closure, &normalized_female_fertility_closure, remove_non_reproducing, male_menopause, female_menopause)
    }
}