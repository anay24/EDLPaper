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


# Plane built like: [w, b]
# w is [a, b, c]
# Plane means w^T x + b = 0
# Example: [[1, 0, 0], 0] means x = 0


def dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def cross(u, v):
    return [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]


def solve_two_by_two(a1, b1, c1, a2, b2, c2):
    bottom = a1 * b2 - b1 * a2

    if abs(bottom) < 0.000001:
        return None

    x = (c1 * b2 - b1 * c2) / bottom
    y = (a1 * c2 - c1 * a2) / bottom
    return [x, y]


def get_line_point(plane1, plane2, direction):
    w1 = plane1[0]
    b1 = plane1[1]
    w2 = plane2[0]
    b2 = plane2[1]

    # If direction[0] is not zero, then the y-z part can be solved.
    # So just set x = 0 and solve for y and z.
    if abs(direction[0]) > 0.000001:
        answer = solve_two_by_two(w1[1], w1[2], -b1, w2[1], w2[2], -b2)
        if answer is None:
            return None
        y = answer[0]
        z = answer[1]
        return [0, y, z]

    # If direction[1] is not zero, set y = 0 and solve for x and z.
    if abs(direction[1]) > 0.000001:
        answer = solve_two_by_two(w1[0], w1[2], -b1, w2[0], w2[2], -b2)
        if answer is None:
            return None
        x = answer[0]
        z = answer[1]
        return [x, 0, z]

    # Otherwise set z = 0 and solve for x and y.
    answer = solve_two_by_two(w1[0], w1[1], -b1, w2[0], w2[1], -b2)
    if answer is None:
        return None

    x = answer[0]
    y = answer[1]
    return [x, y, 0]


def threeDP2Helper(plane1, plane2, radius):
    if radius < 0:
        return []

    w1 = plane1[0]
    w2 = plane2[0]

    direction = cross(w1, w2)

    if dot(direction, direction) < 0.000001:
        return []

    line_point = get_line_point(plane1, plane2, direction)
    if line_point is None:
        return []

    px = line_point[0]
    py = line_point[1]
    pz = line_point[2]

    dx = direction[0]
    dy = direction[1]
    dz = direction[2]

    # A point on the line looks like:
    # [px + t*dx, py + t*dy, pz + t*dz]
    # Plug that into the sphere equation.
    A = dx * dx + dy * dy + dz * dz
    B = 2 * (px * dx + py * dy + pz * dz)
    C = px * px + py * py + pz * pz - radius * radius

    discriminant = B * B - 4 * A * C

    if discriminant < -0.000001:
        return []

    if discriminant < 0:
        discriminant = 0

    root = math.sqrt(discriminant)

    t1 = (-B + root) / (2 * A)
    point1 = [px + t1 * dx, py + t1 * dy, pz + t1 * dz]

    if root < 0.000001:
        return [point1]

    t2 = (-B - root) / (2 * A)
    point2 = [px + t2 * dx, py + t2 * dy, pz + t2 * dz]

    return [point1, point2]


def close(a, b):
    return abs(a - b) < 0.000001


def same_point(p, q):
    return close(p[0], q[0]) and close(p[1], q[1]) and close(p[2], q[2])


def has_point(points, target):
    for point in points:
        if same_point(point, target):
            return True

    return False


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


def test_3d_two_points():
    plane1 = [[1, 0, 0], 0]
    plane2 = [[0, 1, 0], 0]

    points = threeDP2Helper(plane1, plane2, 2)

    assert len(points) == 2
    assert has_point(points, [0, 0, 2])
    assert has_point(points, [0, 0, -2])


def test_3d_one_point():
    plane1 = [[1, 0, 0], -1]
    plane2 = [[0, 1, 0], 0]

    points = threeDP2Helper(plane1, plane2, 1)

    assert len(points) == 1
    assert same_point(points[0], [1, 0, 0])


def test_3d_no_point():
    plane1 = [[1, 0, 0], -2]
    plane2 = [[0, 1, 0], 0]

    points = threeDP2Helper(plane1, plane2, 1)

    assert points == []


def test_3d_parallel_planes():
    plane1 = [[1, 0, 0], 0]
    plane2 = [[2, 0, 0], 1]

    points = threeDP2Helper(plane1, plane2, 1)

    assert points == []


test_yes()
test_no()
test_3d_two_points()
test_3d_one_point()
test_3d_no_point()
test_3d_parallel_planes()
print("ALL TESTS PASSED")
