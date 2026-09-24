"""
tasks/bbh/config.py
BIG-Bench Hard — multi-step symbolic and logical reasoning (date arithmetic, object tracking,
logical deduction, navigation, ...). Every sub-task is cast as options (A), (B), ...; the binary
sub-tasks get (A)/(B) for Yes/No, True/False or Valid/Invalid.
Data: tasks/bbh/gen_bbh.py (23 sub-tasks, balanced).
"""

from ..choice import ChoiceBenchmark
from ..registry import register


@register("bbh")
class BBHConfig(ChoiceBenchmark):
    NAME = "bbh"

    NOTE = (
        "BIG-Bench Hard tests multi-step symbolic and logical reasoning: date arithmetic, object "
        "tracking, spatial navigation, causal judgement and logical deduction. Errors usually come "
        "from skipped steps or losing track of a state (position, count, order, owner)."
    )

    SOLVER_EXAMPLES = """Examples (track the state after every step):

Q: Today is 01/25/2015. What is the date one week from today in MM/DD/YYYY?
(A) 02/01/2015 (B) 01/25/2015 (C) 02/08/2015 (D) 01/18/2015
{"intent":"date arithmetic","reasoning":["Today: 01/25/2015","One week = 7 days","January has 31 days: 25 + 7 = 32 -> February 1","Date: 02/01/2015"],"action":"A","confidence":0.98,"content":"02/01/2015"}

Q: If you follow these instructions, do you return to the starting point? Take 3 steps forward. Take 3 steps backward.
(A) Yes (B) No
{"intent":"navigation","reasoning":["Start at 0","3 steps forward: 0 + 3 = 3","3 steps backward: 3 - 3 = 0","Position 0 is the start"],"action":"A","confidence":0.99,"content":"Yes, back at the start"}

Q: Alice has a red ball, Bob has a blue ball, Claire has a green ball. Alice and Bob swap balls. Then Bob and Claire swap balls. What ball does Bob have?
(A) red ball (B) blue ball (C) green ball
{"intent":"object tracking","reasoning":["Start: Alice=red, Bob=blue, Claire=green","Alice<->Bob: Alice=blue, Bob=red, Claire=green","Bob<->Claire: Alice=blue, Bob=green, Claire=red","Bob has the green ball"],"action":"C","confidence":0.97,"content":"Bob has the green ball"}
"""

    CRITIC_GUIDE = """You are checking multi-step symbolic/logical reasoning.

DISAGREE if:
- A state-tracking step is wrong (miscount, lost position/order/owner)
- A deduction does not follow from the premises
- Date/time arithmetic has an off-by-one or calendar error

AGREE only if:
- The final option matches yours AND every step you can check is correct

When you DISAGREE, name the exact step where the state or logic broke down.
"""

    EXTRA_RULES = (
        "- Write EVERY step, especially state changes (positions, counts, orderings)\n"
        "- Do NOT skip steps, even obvious ones\n"
    )

    FLIP_KEYWORDS = (
        "miscounted", "lost track", "wrong step", "incorrect", "error", "should be",
        "does not follow", "off-by-one", "off by one", "wrong order", "wrong position",
        "contradicts", "misidentif",
    )
