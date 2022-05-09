# from cadcad.spaces import Space
# from cadcad.dynamics import Block
from inspect import Parameter
import numpy as np
from math import sqrt
from random import choice

from typing import Generator
from dataclasses import dataclass

UP = np.array([0, 1])
DOWN = np.array([0, -1])
LEFT = np.array([1, 0])
RIGHT = np.array([-1, 0])


@dataclass
class Hotels():
    uuid: int
    location: np.ndarray
    price: float


@dataclass
class ConsumerState():
    hotel_grid: np.ndarray
    market_share: dict[int, float]
    revenue: dict[int, float]


@dataclass
class WorldState():
    hotels: list[Hotels]
    market_share: ConsumerState


@dataclass
class WorldParams():
    world_shape: tuple[int, int]


def compute_market_share_grid(n_X: int,
                              n_Y: int,
                              hotels: list[Hotels]) -> np.ndarray:
    hotel_grid = np.zeros([n_X, n_Y])

    # Iterate on each grid point and compute the best hotel using an
    # min(distance * price) criteria.
    for x in range(n_X):
        for y in range(n_Y):
            lower_sum = np.inf
            winner_hotel = None
            for hotel in hotels:
                # Euclidean norm
                dx = x - hotel.location[0]
                dy = y - hotel.location[1]
                distance = sqrt(dx ** 2 + dy ** 2)
                current_sum = distance + hotel.price
                if current_sum < lower_sum:
                    lower_sum = current_sum
                    winner_hotel = hotel.uuid
                elif current_sum == lower_sum:
                    winner_hotel = choice([hotel.uuid, winner_hotel])
                else:
                    pass
            hotel_grid[x, y] = winner_hotel
    return hotel_grid


def compute_market_share(hotel_grid: np.ndarray) -> dict[int, float]:
    uuid_frequency = np.unique(hotel_grid, return_counts=True)
    return dict(zip(*uuid_frequency))


def compute_revenues(hotels: list[Hotels],
                     market_shares: dict[int, float]) -> dict[int, float]:
    return {hotel.uuid: market_shares[hotel.uuid] * hotel.price
            if hotel.uuid in market_shares
            else 0.0
            for hotel in hotels}


def realize_consumption(hotels: list[Hotels],
                        world_params: WorldParams) -> ConsumerState:
    n_X = world_params.world_shape[0]
    n_Y = world_params.world_shape[1]

    hotel_grid = compute_market_share_grid(n_X, n_Y, hotels)
    market_shares = compute_market_share(hotel_grid)
    revenues = compute_revenues(hotels, market_shares)

    return ConsumerState(hotel_grid, market_shares, revenues)


def hotel_decisions(world_state: WorldState,
                    world_params: WorldParams) -> list[Hotels]:

    n_X = world_params.world_shape[0]
    n_Y = world_params.world_shape[1]

    new_hotels = world_state.hotels.copy()

    for i, hotel in enumerate(world_state.hotels):
        current_reward = world_state.market_share.revenue[hotel.uuid]

        # Step 1: decide to move or stay
        move = choice([UP, DOWN, LEFT, RIGHT])
        old_location = hotel.location
        new_location = hotel.location + move
        new_location[0] = new_location[0] % n_X
        new_location[1] = new_location[1] % n_Y

        new_hotels[i].location = new_location
        new_world = compute_market_share_grid(n_X, n_Y, new_hotels)
        new_market_share = compute_market_share(new_world)
        hotel_market_share = new_market_share.get(hotel.uuid, 0.0)
        new_reward = hotel_market_share * hotel.price
        if new_reward > current_reward:
            current_reward = new_reward
        else:
            new_hotels[i].location = old_location

        # Step 2: decide to mutate the price
        price_change = choice([-1, 1])
        old_price = hotel.price
        new_price = max(hotel.price + price_change, 0)
        new_hotels[i].price = new_price
        new_world = compute_market_share_grid(n_X, n_Y, new_hotels)
        new_market_share = compute_market_share(new_world)
        hotel_market_share = new_market_share.get(hotel.uuid, 0.0)
        new_reward = hotel_market_share * new_price
        if new_reward > current_reward:
            current_reward = new_reward
        else:
            new_hotels[i].price = old_price
    return new_hotels


@dataclass
class HotellingLawModel():
    hotels: list[Hotels]
    consumers: ConsumerState
    world_params: WorldParams

    @property
    def world_state(self) -> WorldState:
        return WorldState(self.hotels, self.consumers)

    def step(self) -> None:
        self.consumers = realize_consumption(self.hotels, self.world_params)
        self.hotels = hotel_decisions(self.world_state, self.world_params)

    def run(self, n_steps: int) -> Generator[WorldState, None, None]:
        yield self.world_state
        for _ in range(n_steps):
            self.step()
            yield self.world_state


# Legacy cadCAD Model Building Workflow
# 1. Specify state components & parameters
# 2. Specify the model structure / DAG
# 3. Specify the model logic
# 4. Specify the model execution (incl. param and state values)

# 1a. Spaces

Space = callable[str, dict]
Block = callable[callable]

HotelsSpace = Space('Hotels State',
                    dict(hotels=list[Hotels]))

MarketShare = Space('Market Share',
                    dict(market_share=ConsumerState))

# 1b. Parameters

ModelParameters = Space('Parameters',
                        dict(world_shape=tuple[int, int]))

# 2.


# 2 with type hinting on the functions
realize_consumption_block = Block(realize_consumption)
hotel_decisions_block = Block(hotel_decisions)

# 2 with explicit spaces
realize_consumption_block: Block = Block(realize_consumption,
                                         input=HotelsSpace,
                                         output=MarketShare)

hotel_decisions_block: Block = Block(hotel_decisions_block,
                                     input=HotelsSpace + MarketShare
                                     output=HotelsSpace)

model_block: Block = (realize_consumption_block >> hotel_decisions_block)

# model_dag = b_1 >> (b_2 & b_3) >> b_4

# 3. Done above

# 4a. evolve one timestep

initial_state = Point(
    WorldState,
    hotels=INITIAL_HOTELS,
    market_share=INITIAL_MARKET_SHARE
)

parameters = Point(
    ModelParameters,
    world_shape=WORLD_SHAPE
)

new_model_state: WorldState = model_block.step(initial_state,
                                   parameters)

# 4b. run for 1000 timesteps, 20 MCs and some sweeps

initial_state_to_sweep = PointList(
    hotels=[None, None],
    market_share=[None, None]
)

parameters_to_sweep = PointList(
    world_shape=[None, None, None]
)

results_without_speed = model_block.simulate(initial_state,
                                             parameters,
                                             steps=1000,
                                             samples=20)

results_with_sweep = model_block.simulate(initial_state_to_sweep,
                                          parameters_to_sweep,
                                          steps=1000,
                                          samples=20)
