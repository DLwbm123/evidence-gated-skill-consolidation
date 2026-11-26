import ast
import json
from pathlib import Path
import re
import runpy

import torch
import textworld
import textworld.gym
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
from transformers import AutoModelForCausalLM, AutoTokenizer


def author_constant(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"missing {name} in author source")


def main():
    torch.manual_seed(7)
    source = Path("source/f")
    artifact = Path("models/m0")
    trainer = source / "verl/trainer/ppo"
    loss_util = runpy.run_path(str(trainer / "oprd_utils.py"))
    proj_util = runpy.run_path(str(trainer / "oprd_projector.py"))
    instruction = author_constant(trainer / "pr_opd_utils.py", "SKILL_ANALYSIS_INSTRUCTION")
    prompt_template = author_constant(
        source / "agent_system/environments/prompts/alfworld.py", "ALFWORLD_TEMPLATE_NO_HIS"
    )
    tokenizer = AutoTokenizer.from_pretrained(artifact, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        artifact, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    model.eval()

    def prompt_ids(text):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

    def generate(tokens, limit):
        with torch.no_grad():
            return model.generate(
                tokens, max_new_tokens=limit, do_sample=True, temperature=0.8,
                attention_mask=torch.ones_like(tokens),
                pad_token_id=tokenizer.eos_token_id,
            )[:, tokens.shape[-1]:]

    root = Path("a-protocol")
    rows = json.loads((root / "families.json").read_text())
    game = root / rows["splits"]["train"]["Pick"][0]
    environment_id = textworld.gym.register_games(
        [str(game.resolve())],
        textworld.EnvInfos(won=True, admissible_commands=True),
        batch_size=1,
        asynchronous=False,
        max_episode_steps=8,
        wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
    )
    env = textworld.gym.make(environment_id)
    turns = []
    try:
        observations, info = env.reset()
        for _ in range(8):
            prompt = prompt_template.format(
                current_observation=observations[0],
                admissible_actions=", ".join(info["admissible_commands"][0]),
            )
            response = generate(prompt_ids(prompt), 96)
            text = tokenizer.decode(response[0], skip_special_tokens=True)
            match = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
            action = match.group(1).strip() if match else text.strip()
            turns.append((prompt, response, text))
            observations, _, done, info = env.step([action])
            if done[0]:
                break
        won = bool(info["won"][0])
    finally:
        env.close()

    trajectory = "\n".join(f"{p[:850]}\nResponse: {text}" for p, _, text in turns)
    hindsight = f"{trajectory}\nOutcome: {'SUCCEEDED' if won else 'FAILED'}\n{instruction}"
    skill_ids = generate(prompt_ids(hindsight), 128)
    skill = tokenizer.decode(skill_ids[0], skip_special_tokens=True).strip()
    assert skill, "the same model returned an empty hindsight skill"

    prompt, response_ids, _ = turns[-1]
    student_prompt = prompt_ids(prompt)
    teacher_prompt = prompt_ids(f"[Hindsight Skill Information]\n{skill}\n\n{prompt}")
    student_input = torch.cat((student_prompt, response_ids), dim=1)
    teacher_input = torch.cat((teacher_prompt, response_ids), dim=1)
    count = response_ids.shape[1]
    assert count > 0 and torch.equal(student_input[:, -count:], teacher_input[:, -count:])
    assert student_prompt.shape[1] != teacher_prompt.shape[1]

    layers = loss_util["resolve_layer_spec"]("all", model.config.num_hidden_layers)
    projectors = proj_util["OPRDProjectors"](layers, model.config.hidden_size, hidden=512).cuda()
    with torch.no_grad():
        teacher_output = model(teacher_input, output_hidden_states=True, use_cache=False)
    student_output = model(student_input, output_hidden_states=True, use_cache=False)

    def response_states(output, prompt_length):
        start = prompt_length - 1
        return {layer: output.hidden_states[layer][:, start:start + count] for layer in layers}

    teacher = response_states(teacher_output, teacher_prompt.shape[1])
    student = response_states(student_output, student_prompt.shape[1])
    projected = {layer: projectors(layer, state) for layer, state in student.items()}
    mask = torch.ones_like(response_ids)
    loss, metrics = loss_util["compute_oprd_loss"](projected, teacher, mask)
    assert torch.isfinite(loss) and loss.item() > 0
    assert not any(t.requires_grad for t in teacher.values())
    (0.01 * loss).backward()
    actor_gradient = model.model.layers[0].self_attn.q_proj.weight.grad
    projector_gradient = next(projectors.parameters()).grad
    for gradient in (actor_gradient, projector_gradient):
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().max() > 0
    optimizer = torch.optim.AdamW(projectors.parameters(), lr=1e-4)
    before = next(projectors.parameters()).detach().clone()
    optimizer.step()
    assert not torch.equal(before, next(projectors.parameters()))
    print(json.dumps({
        "turns": len(turns), "episode_won": won, "skill_tokens": skill_ids.shape[1],
        "aligned_response_tokens": count, "actor_gradient_nonzero": True,
        "projector_updated": True, "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        **metrics,
    }), flush=True)


if __name__ == "__main__":
    main()
