from cog import BaseModel, Input, Path


class TrainingOutput(BaseModel):
    weights: Path


def train(
    prefix: str = Input(description="Prefix for inference model"),
) -> TrainingOutput:
    output_path = Path("/tmp/out.txt")
    output_path.write_text(prefix)
    return TrainingOutput(weights=output_path)
