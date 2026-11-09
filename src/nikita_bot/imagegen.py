import subprocess
import tempfile
from pathlib import Path


class StableDiffusionCpp:
    def __init__(self, executable_path: str, model_path: str) -> None:
        self.executable_path = Path(executable_path)
        self.model_path = Path(model_path)

    def generate(self, prompt: str) -> bytes:
        if not self.executable_path.is_file():
            raise FileNotFoundError(f"Не найден sd-cli: {self.executable_path}")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Не найдена модель SD-Turbo: {self.model_path}")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "generated.png"
            result = subprocess.run(
                [
                    str(self.executable_path),
                    "-m",
                    str(self.model_path),
                    "-p",
                    prompt,
                    "-o",
                    str(output_path),
                    "-W",
                    "512",
                    "-H",
                    "512",
                    "--steps",
                    "8",
                    "--cfg-scale",
                    "1",
                    "--sampling-method",
                    "euler",
                    "--backend",
                    "vulkan0",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=900,
            )
            if result.returncode != 0:
                details = (result.stderr or result.stdout)[-4000:]
                raise RuntimeError(f"sd-cli завершился с ошибкой:\n{details}")
            if not output_path.is_file():
                raise RuntimeError("sd-cli завершился без выходного PNG")
            return output_path.read_bytes()