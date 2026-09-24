"""
tasks/commonsenseqa/config.py
CommonsenseQA — everyday commonsense knowledge, 5 options (A-E), one correct.
Data: tasks/commonsenseqa/gen_commonsenseqa.py (validation split, 1,221 questions with labels).
"""

from ..choice import ChoiceBenchmark
from ..registry import register


@register("commonsenseqa")
class CommonsenseQAConfig(ChoiceBenchmark):
    NAME = "commonsenseqa"

    NOTE = (
        "CommonsenseQA tests everyday commonsense knowledge: real-world situations, cause and "
        "effect, social norms and general world knowledge. Exactly one of the 5 options is correct."
    )

    SOLVER_EXAMPLES = """Examples:

Q: Where would you find a penguin in its natural habitat?
A) Desert  B) Jungle  C) Antarctic  D) Savanna  E) Canadian tundra
{"intent":"geography","reasoning":["Penguins are native to the Southern Hemisphere","Most species live in and around Antarctica","The Canadian tundra is Arctic, where there are no wild penguins"],"action":"C","confidence":0.99,"content":"Penguins live in the Antarctic"}

Q: What do people typically do when they feel cold?
A) Swim  B) Remove clothing  C) Put on more clothing  D) Open windows  E) Eat ice cream
{"intent":"everyday","reasoning":["Feeling cold calls for more insulation","More clothing keeps body heat in","The other options make a person colder"],"action":"C","confidence":0.99,"content":"They put on more clothing"}

Q: If you want to send a letter overseas, what do you need?
A) A telephone  B) A postage stamp  C) A computer  D) A fax machine  E) A radio
{"intent":"everyday","reasoning":["A physical letter goes through the postal service","Postal delivery requires postage","The other options are for electronic communication"],"action":"B","confidence":0.98,"content":"A postage stamp"}
"""

    CRITIC_GUIDE = """DISAGREE if:
- The chosen option contradicts everyday knowledge
- The reasoning contains a clear logical or factual error
- A clearly better option exists among the choices

AGREE only if:
- The option fits common sense and the reasoning applies everyday logic correctly

Do NOT disagree over phrasing when the chosen letter is the same.
"""

    EXTRA_RULES = "- Use everyday logic and world knowledge; pick the MOST typical option\n"
