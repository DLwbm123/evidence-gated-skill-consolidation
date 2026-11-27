"""Fixed, authored Writing continuations. Secondary teacher-forced style diagnostic."""
import torch
from torch.nn import functional as F

# Each tuple is a prompt followed by Cute, Classical and Chuunibyou continuations.
REFERENCES = (
    ("A clockmaker finds a clock that runs backward.",
     "Grandpa tucked the clock into a woolly sock. It tickled his thumb and gave him yesterday's biscuit back.",
     "The clockmaker laid the instrument on the bench. Its retreating hand marked an hour he had long regretted.",
     "The backward hand had chosen me. Tonight I would break the iron oath of the Lords of Time."),
    ("A letter slips under the lighthouse door.",
     "Pip opened the damp envelope with one careful paw. Inside was a tiny drawing of their lighthouse, complete with whiskers.",
     "The keeper recognized the handwriting before he broke the seal. He put down his cup and drew the lamp nearer.",
     "At last the abyss had written back. Its black seal burned like the mark I carried beneath my glove."),
    ("Someone has left a spare key on the garden wall.",
     "A robin sat beside the key as if minding a shop. Nell paid with two crumbs and received a cheerful chirp.",
     "Rain had washed the iron clean. She took the key in silence and followed the wall to the western gate.",
     "The seal lay broken on the wall. Let the seven wardens tremble, for their prisoner now held the key."),
    ("The last train stops in an unfamiliar village.",
     "A cat in a little scarf greeted every passenger. On the platform a kettle was singing just for them.",
     "No name was painted on the platform. The travelers waited in the rain while the guard consulted his watch.",
     "Beyond the platform stood the village erased from every map. Only I could hear the names its dead were chanting."),
    ("A gardener hears a voice from an empty flowerpot.",
     "The pot asked for a blanket. Mae tucked moss around its rim and promised a very small bedtime story.",
     "He set aside the spade and bent toward the clay. The voice spoke again, with the measured accent of his father.",
     "From its hollow throne the buried monarch whispered my forbidden title. Even the thorns bowed as I answered."),
    ("A snowstorm uncovers a wooden bridge.",
     "Two rabbits crossed first, leaving little dimpled footprints. Bea laid a mitten on the rail like a tiny flag.",
     "At dawn the old bridge stood clear of the drift. Its weathered timbers still bore the mason's mark.",
     "The bridge of exile had risen from the white void. Across its planks waited the shadow that wore my face."),
    ("A baker opens the oven and finds a map.",
     "The map smelled faintly of cinnamon. A raisin marked home, so Jo gave it a happy little pat.",
     "The baker spread the curling sheet on the table. In the margin someone had traced the river as it ran before the flood.",
     "The ember map revealed the road to the Ashen Crown. No mortal baker should have survived the sight of it."),
    ("Every umbrella on the street opens at once.",
     "The umbrellas bobbed like friendly mushrooms. A little boy offered his blue one a biscuit for keeping him dry.",
     "Along the avenue the canopies rose in silence. From the upstairs window she watched the slow procession pass.",
     "A hundred black wings unfurled at my command. The city had mistaken my gathering legion for a shower of rain."),
)


@torch.inference_mode()
def measure(actor, tokenizer, template):
    was_training = actor.training
    actor.eval()
    result = []
    try:
        for idx, (prompt, *continuations) in enumerate(REFERENCES):
            user = template.replace("{task_description}", prompt)
            prefix = tokenizer.apply_chat_template([dict(role="user", content=user)], add_generation_prompt=True)
            scores = []
            for passage in continuations:
                suffix = tokenizer.encode(passage, add_special_tokens=False)
                ids = torch.tensor([prefix + suffix], device=actor.device)
                logits = actor(ids, use_cache=False, logits_to_keep=len(suffix)+1).logits[0, :-1].float()
                scores.append(-F.cross_entropy(logits, ids[0, len(prefix):]).item())
            result.append(dict(item=idx+1, cute=scores[0], classical=scores[1], chuunibyou=scores[2],
                               margin=scores[0]-(scores[1]+scores[2])/2))
    finally:
        actor.train(was_training)
    return result
