from .exceptions import CogSafePushError
from .task_context import TaskContext


def handle_deployment(task_context: TaskContext, version: str) -> None:
    """Create or update a deployment for the model."""
    if not task_context.deployment_name:
        return

    deployment_name = task_context.deployment_name
    deployment_owner = task_context.deployment_owner or task_context.model.owner

    try:
        # Check if deployment exists
        current_deployment = task_context.client.deployments.get(
            f"{deployment_owner}/{deployment_name}"
        )
        update_deployment(task_context, current_deployment, version)
    except Exception as e:
        if "not found" in str(e).lower():
            create_deployment(task_context, version)
        else:
            raise CogSafePushError(f"Failed to check deployment: {str(e)}")


def create_deployment(task_context: TaskContext, version: str) -> None:
    """Create a new deployment for the model."""
    deployment_name = task_context.deployment_name
    hardware = task_context.deployment_hardware or "cpu"
    if hardware == "cpu":
        min_instances = 1
        max_instances = 20
    else:
        min_instances = 0
        max_instances = 2

    print(
        f"Creating deployment {deployment_name} with {hardware} hardware, {min_instances} min instances, {max_instances} max instances"
    )

    try:
        task_context.client.deployments.create(
            name=deployment_name,
            model=f"{task_context.model.owner}/{task_context.model.name}",
            version=version,
            hardware=hardware,
            min_instances=min_instances,
            max_instances=max_instances,
        )
    except Exception as e:
        raise CogSafePushError(f"Failed to create deployment: {str(e)}")


def update_deployment(
    task_context: TaskContext,
    current_deployment,
    version: str,
) -> None:
    """Update an existing deployment for the model."""
    current_config = current_deployment.current_release.configuration
    print(
        f"Updating deployment {current_deployment.name} with {current_config.hardware} hardware, {current_config.min_instances} min instances, {current_config.max_instances} max instances"
    )
    print(
        f"Changing version from {current_deployment.current_release.version} to {version}"
    )

    try:
        task_context.client.deployments.update(
            deployment_owner=current_deployment.owner,
            deployment_name=current_deployment.name,
            version=version,
        )
    except Exception as e:
        raise CogSafePushError(f"Failed to update deployment: {str(e)}")
