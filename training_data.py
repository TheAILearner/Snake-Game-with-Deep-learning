from game import *

def generate_training_data(display, clock):
    training_data_x = []
    training_data_y = []
    training_games = 1000
    steps_per_game = 2000

    for _ in tqdm(range(training_games)):
        snake_start, snake_position, apple_position, score = starting_positions()
        prev_apple_distance = apple_distance_from_snake(apple_position, snake_position)

        for _ in range(steps_per_game):
            angle, snake_direction_vector, apple_direction_vector_normalized, snake_direction_vector_normalized = angle_with_apple(
                snake_position, apple_position)
            direction, button_direction = generate_random_direction(snake_position, angle)
            current_direction_vector, is_front_blocked, is_left_blocked, is_right_blocked = blocked_directions(
                snake_position)

            direction, button_direction, training_data_y = generate_training_data_y(snake_position, angle_with_apple,
                                                                                    button_direction, direction,
                                                                                    training_data_y, is_front_blocked,
                                                                                    is_left_blocked, is_right_blocked)

            if is_front_blocked == 1 and is_left_blocked == 1 and is_right_blocked == 1:
                break

            training_data_x.append(
                [is_left_blocked, is_front_blocked, is_right_blocked, apple_direction_vector_normalized[0], \
                 snake_direction_vector_normalized[0], apple_direction_vector_normalized[1], \
                 snake_direction_vector_normalized[1]])

            snake_position, apple_position, score = play_game(snake_start, snake_position, apple_position,
                                                              button_direction, score, display, clock)

    return training_data_x, training_data_y


def generate_training_data_y(snake_position, angle_with_apple, button_direction, direction, training_data_y,
                             is_front_blocked, is_left_blocked, is_right_blocked):
    if direction == -1:
        if is_left_blocked == 1:
            if is_front_blocked == 1 and is_right_blocked == 0:
                direction, button_direction = direction_vector(snake_position, angle_with_apple, 1)
                training_data_y.append([0, 0, 1])
            elif is_front_blocked == 0 and is_right_blocked == 1:
                direction, button_direction = direction_vector(snake_position, angle_with_apple, 0)
                training_data_y.append([0, 1, 0])
            elif is_front_blocked == 0 and is_right_blocked == 0:
                direction, button_direction = direction_vector(snake_position, angle_with_apple, 1)
                training_data_y.append([0, 0, 1])

        else:
            training_data_y.append([1, 0, 0])

    elif direction == 0:
        if is_front_blocked == 1:
            if is_left_blocked == 1 and is_right_blocked == 0:
                direction, button_direction = direction_vector(snake_position, angle_with_apple, 1)
                training_data_y.append([0, 0, 1])
            elif is_left_blocked == 0 and is_right_blocked == 1:
                direction, button_direction = direction_vector(snake_position, angle_with_apple, -1)
                training_data_y.append([1, 0, 0])
            elif is_left_blocked == 0 and is_right_blocked == 0:
                training_data_y.append([0, 0, 1])
                direction, button_direction = direction_vector(snake_position, angle_with_apple, 1)
        else:
            training_data_y.append([0, 1, 0])
    else:
        if is_right_blocked == 1:
            if is_left_blocked == 1 and is_front_blocked == 0:
                direction, button_direction = direction_vector(snake_position, angle_with_apple, 0)
                training_data_y.append([0, 1, 0])
            elif is_left_blocked == 0 and is_front_blocked == 1:
                direction, button_direction = direction_vector(snake_position, angle_with_apple, -1)
                training_data_y.append([1, 0, 0])
            elif is_left_blocked == 0 and is_front_blocked == 0:
                direction, button_direction = direction_vector(snake_position, angle_with_apple, -1)
                training_data_y.append([1, 0, 0])
        else:
            training_data_y.append([0, 0, 1])

    return direction, button_direction, training_data_y