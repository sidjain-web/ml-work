"""Build a small offline text corpus for character-level language modeling.

The text below is a set of classic fables retold in original wording (the
underlying stories are public domain). This gives coherent, varied English —
narrative structure, dialogue, punctuation — which is what a char-level model
needs to learn something visible during a quick training run.

For a larger, "real" corpus use `download_data.py` (needs network).

Run:
    python build_corpus.py            # writes tiny_corpus.txt next to this file
"""
from __future__ import annotations

import os

FABLES = [
    ("The Fox and the Grapes",
     "One warm afternoon a hungry fox padded through an orchard and spotted a "
     "cluster of ripe grapes hanging high on a vine. His mouth watered at once. "
     "He crouched low and sprang, but the grapes hung just out of reach. He "
     "tried again, running and leaping with all his strength, yet each jump fell "
     "short. Again and again he tried, until his legs ached and his breath came "
     "in gasps. At last he gave up and trotted away with his nose in the air. "
     "\"I did not want those grapes anyway,\" he said. \"They are surely sour.\" "
     "It is easy to despise what you cannot have."),

    ("The Tortoise and the Hare",
     "A hare was forever boasting about how swift he was. Tired of his bragging, "
     "a tortoise finally challenged him to a race. The hare laughed, for the "
     "notion seemed absurd, but he agreed. When the race began the hare bounded "
     "far ahead, and seeing the tortoise plodding slowly behind, he decided to "
     "rest beneath a shady tree. He soon fell fast asleep. The tortoise never "
     "stopped. Step by patient step he crept along the road, and by the time the "
     "hare awoke and dashed to the finish, the tortoise had already crossed the "
     "line. Slow and steady wins the race."),

    ("The Ant and the Grasshopper",
     "All summer long a grasshopper sang and danced in the meadow, while the ants "
     "worked without rest, carrying grain to their nest. \"Why toil so hard?\" the "
     "grasshopper laughed. \"Come and sing with me instead.\" The ants only "
     "answered that they were storing food for the winter, and they advised him "
     "to do the same. But the grasshopper waved them off and kept on playing. "
     "When winter came the fields were bare and cold. The grasshopper, hungry and "
     "shivering, found the ants comfortable and well fed. Too late he understood "
     "that there is a time for work and a time for play."),

    ("The Lion and the Mouse",
     "A mighty lion lay sleeping when a little mouse ran across his paw and woke "
     "him. The lion seized the tiny creature and prepared to eat him. \"Please "
     "spare me,\" squeaked the mouse, \"and one day I may repay your kindness.\" "
     "The lion laughed at the thought that so small a thing could ever help him, "
     "but he let the mouse go. Some days later hunters trapped the lion in a "
     "strong net, and he roared in helpless fury. The little mouse heard him and "
     "hurried to the spot. Patiently he gnawed the ropes until the lion was free. "
     "No act of kindness, however small, is ever wasted."),

    ("The Boy Who Cried Wolf",
     "A shepherd boy grew bored watching his flock on the quiet hillside, so for "
     "amusement he cried out, \"Wolf! Wolf!\" The villagers came running to help, "
     "only to find him laughing at the trick. He played the same joke a second "
     "time, and again they rushed up the hill for nothing. Then one day a wolf "
     "truly came among the sheep. The boy screamed for help with all his might, "
     "but the villagers, sure it was another lie, stayed in their homes. The wolf "
     "scattered the flock at his leisure. Nobody believes a liar, even when he "
     "speaks the truth."),

    ("The Crow and the Pitcher",
     "A thirsty crow found a pitcher with a little water at the bottom, but the "
     "neck was too narrow for his beak to reach. He pushed and strained, yet "
     "could not tip the heavy pitcher over. Then a clever idea struck him. One by "
     "one he dropped small pebbles into the pitcher, and with each stone the water "
     "rose a little higher. He kept at it, patient and steady, until at last the "
     "water reached the brim and he drank his fill. Little by little, thoughtful "
     "effort can solve what force alone cannot."),

    ("The Goose and the Golden Egg",
     "A farmer once owned a goose that laid a single egg of pure gold every "
     "morning. Day by day he grew richer, yet the more he had, the more he "
     "wanted. Impatient with one egg a day, he reasoned that the goose must be "
     "full of gold inside, and he killed it to take the whole treasure at once. "
     "But when he cut the goose open he found it was just like any other. There "
     "was no gold within, and now there would be no golden egg tomorrow. Greed "
     "often destroys the very thing that feeds it."),

    ("The North Wind and the Sun",
     "The north wind and the sun argued over which of them was stronger. To "
     "settle it, they agreed that whoever could make a passing traveler remove "
     "his cloak would be the winner. The north wind blew first, cold and fierce, "
     "but the harder he blew, the tighter the traveler wrapped the cloak around "
     "himself. At last the wind gave up. Then the sun shone gently, warming the "
     "air little by little. Soon the traveler grew hot, loosened his cloak, and "
     "finally took it off altogether. Gentle warmth achieves what raw force "
     "cannot."),

    ("The Dog and His Reflection",
     "A dog was carrying a fine bone home across a narrow bridge over a stream. "
     "Looking down, he saw his own reflection in the water and mistook it for "
     "another dog with a bone of its own. Greedy for both, he snapped at the "
     "reflection to seize the second bone. The moment he opened his mouth, his "
     "own bone dropped into the stream and sank out of sight. He was left with "
     "nothing. Grasp at more than your share and you may lose what you already "
     "hold."),

    ("The Milkmaid and Her Pail",
     "A milkmaid walked to market balancing a pail of milk upon her head. As she "
     "went she dreamed of all she would do. With the money from the milk she "
     "would buy eggs; the eggs would hatch into chickens; the chickens she would "
     "sell to buy a fine new gown. In that gown, she imagined, she would toss her "
     "head proudly at the fair. Lost in the daydream, she tossed her head for "
     "real. Down came the pail, and the milk spilled across the road, and with it "
     "went every castle in the air. Do not count your riches before you have "
     "them."),
]

HEADER = (
    "A COLLECTION OF FABLES\n"
    "Classic tales retold, for character-level language modeling.\n"
    "\n"
)


def build(repeat: int = 6) -> str:
    """Assemble the corpus. `repeat` cycles the fables so the file is large
    enough for a quick model to train on; each cycle is identical text, which is
    fine for a demo. Use download_data.py for a larger, non-repeating corpus."""
    parts = [HEADER]
    for _ in range(repeat):
        for title, body in FABLES:
            parts.append(title.upper())
            parts.append("")
            parts.append(body)
            parts.append("")
            parts.append("")
    return "\n".join(parts)


if __name__ == "__main__":
    text = build()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_corpus.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {out_path}")
    print(f"characters: {len(text):,}")
    print(f"unique chars: {len(set(text))}")
