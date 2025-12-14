# -*- coding: utf-8 -*-
from clearml import Task, OutputModel
from clearml.automation import HyperParameterOptimizer, UniformParameterRange, UniformIntegerParameterRange
from clearml.automation.optuna import OptimizerOptuna

PROJECT_NAME = "Weather_Forecast_Project"

def main():
    task = Task.init(
        project_name=PROJECT_NAME,
        task_name="HPO_Controller",
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False
    )

    #берём последнюю базовую обучающую задачу
    available_tasks = Task.get_tasks(
        project_name=PROJECT_NAME,
        task_name="Train_Model_Base",
        task_filter={'status': ['completed', 'published']}
    )

    if not available_tasks:
        print("Base task 'Train_Model_Base' not found. Run training first.")
        return

    available_tasks.sort(key=lambda t: t.data.created, reverse=True)
    base_task_id = available_tasks[0].id
    print(f"Using base task ID: {base_task_id}")

    optimizer = HyperParameterOptimizer(
        base_task_id=base_task_id,
        hyper_parameters=[
            UniformIntegerParameterRange("General/depth", min_value=4, max_value=10, step_size=1),
            UniformParameterRange("General/learning_rate", min_value=0.01, max_value=0.3, step_size=0.01),
            UniformParameterRange("General/l2_leaf_reg", min_value=1.0, max_value=10.0, step_size=0.5),
        ],
        objective_metric_title="Metrics",
        objective_metric_series="MAE",
        objective_metric_sign="min",
        max_number_of_concurrent_tasks=1,  # можно увеличить, если очередь позволяет
        optimizer_class=OptimizerOptuna,
        execution_queue="default",
        total_max_jobs=5,
        max_iteration_per_job=30,
        pool_period_min=0.2
    )

    print("Starting HPO...")
    optimizer.start_locally()
    optimizer.wait()

    #берём лучшую задачу
    top_exp = optimizer.get_top_experiments(top_k=1)
    if top_exp:
        best_task = top_exp[0]
        print(f"Best Task ID: {best_task.id}")
        mae = best_task.get_last_scalar_metrics().get("Metrics", {}).get("MAE", {}).get("last")
        print(f"Best MAE: {mae}")

        best_model_artifact = best_task.artifacts.get("model")
        if best_model_artifact:
            output_model = OutputModel(task=task, name="weather_predictor_model")
            output_model.update_weights(register_uri=best_model_artifact.url)
            output_model.publish()
            print("Best model registered and published.")
        else:
            print("Could not find model artifact in best task.")
    else:
        print("No experiments completed.")

    task.close()

if __name__ == "__main__":
    main()
