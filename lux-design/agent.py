import math, sys
from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate

DIRECTIONS = Constants.DIRECTIONS
game_state = None


# lux-ai-2021 main.py main.py --out=replays\replay.json


def get_resource_tiles(game_state) -> list[Cell]:
    return [
        game_state.map.get_cell(x, y) for y in range(game_state.map.height) for x in range(game_state.map.width) if game_state.map.get_cell(x, y).has_resource()
    ]


def get_closest_resource_tile(player, unit, resource_tiles):
    closest_dist = math.inf
    closest_resource_tile = None
    for resource_tile in resource_tiles:
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.COAL and not player.researched_coal(): continue
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.URANIUM and not player.researched_uranium(): continue
        dist = resource_tile.pos.distance_to(unit.pos)
        if dist < closest_dist:
            closest_dist = dist
            closest_resource_tile = resource_tile
    return closest_resource_tile


def get_closest_city_tile(player, unit):
    closest_dist = math.inf
    closest_city_tile = None
    for k, city in player.cities.items():
        for city_tile in city.citytiles:
            dist = city_tile.pos.distance_to(unit.pos)
            if dist < closest_dist:
                closest_dist = dist
                closest_city_tile = city_tile
    return closest_city_tile


def agent(observation, configuration):
    global game_state
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])
    actions = []
    player = game_state.players[observation.player]
    opponent = game_state.players[(observation.player + 1) % 2]
    cities_tiles = [
        city_tile for city in player.cities.values() for city_tile in city.citytiles
    ]
    resource_tiles: list[Cell] = get_resource_tiles(game_state)
    for unit in player.units:
        if unit.is_worker() and unit.can_act():
            if unit.get_cargo_space_left() > 0:
                closest_resource_tile = get_closest_resource_tile(player, unit, resource_tiles)
                if closest_resource_tile is not None:
                    actions.append(unit.move(unit.pos.direction_to(closest_resource_tile.pos)))
            else:
                if len(player.cities) > 0:
                    closest_city_tile = get_closest_city_tile(player, unit)
                    if closest_city_tile is not None:
                        move_dir = unit.pos.direction_to(closest_city_tile.pos)
                        actions.append(unit.move(move_dir))

    # you can add debug annotations using the functions in the annotate object
    # actions.append(annotate.circle(0, 0))

    return actions
