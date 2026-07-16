from logic import *

rain = Symbol("rain")  # It rained.
pool = Symbol("pool")  # Bob went to the pool.
movies = Symbol("movies")  # Bob went to the movies.

knowledge = And(
    Implication(Not(rain), pool),  # If it did not rain, bob went to the pool.
    Or(pool, movies),              # Bob went to either the pool or the movies,
    Not(And(pool, movies)),        # but not both.
    movies                         # Bob went to the movies.
)

print("Did Bob go to the pool?", "Yes" if model_check(knowledge, pool) else "No")
print("Did it rain?", "Yes" if model_check(knowledge, rain) else "No")
