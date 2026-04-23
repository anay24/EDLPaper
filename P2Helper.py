import math

# Inequality built like: [a, b, sign, c]
# Standsfor: a*x + b*y sign c
# Example: [1, 0, ">=", 0] means x >= 0

def get_points(line, radius):
    a = line[0]
    b = line[1]
    c = line[3]

    points = []

    # Line is a*x = c meaning  x = c/a
    if b == 0:
        if a == 0:
            return points

        x = c / a
        y_squared = radius * radius - x * x

        if y_squared < 0:
            return points

        y = math.sqrt(y_squared)
        points.append([x, y])

        if y != 0:
            points.append([x, -y])

        return points

    # Else solve for y:
    # a*x + b*y = c
    # y = (c - a*x) / b
    #  plug into:
    # x^2 + y^2 = rho^2
    A = a * a + b * b
    B = -2 * a * c
    C = c * c - radius * radius * b * b

    discriminant = B * B - 4 * A * C

    if discriminant < 0:
        return points

    x1 = (-B + math.sqrt(discriminant)) / (2 * A)
    y1 = (c - a * x1) / b
    points.append([x1, y1])

    if discriminant != 0:
        x2 = (-B - math.sqrt(discriminant)) / (2 * A)
        y2 = (c - a * x2) / b
        points.append([x2, y2])

    return points


def works(point, inequalities):
    x = point[0]
    y = point[1]

    for inequality in inequalities:
        a = inequality[0]
        b = inequality[1]
        sign = inequality[2]
        c = inequality[3]

        left_side = a * x + b * y

        if sign == "<=" and left_side > c + 0.000001:
            return False

        if sign == ">=" and left_side < c - 0.000001:
            return False

    return True


def find_point(inequalities, radius):
    all_points = []

    for inequality in inequalities:
        points = get_points(inequality, radius)

        for point in points:
            all_points.append(point)

    for point in all_points:
        if works(point, inequalities):
            return point

    return None


def test_yes():
    inequalities = [
        [1, 0, ">=", 0],  # x >= 0
        [0, 1, ">=", 0],  # y >= 0
    ]

    point = find_point(inequalities, 1)

    assert point is not None
    assert works(point, inequalities)


def test_no():
    inequalities = [
        [1, 0, "<=", -0.5],  # x <= -0.5
        [1, 0, ">=", 0.5],   # x >= 0.5
    ]

    point = find_point(inequalities, 1)

    assert point is None


test_yes()
test_no()
print("CODE EXECUTION COMPLETE")
