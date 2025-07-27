# ---------- Geometry ----------
class Box:
    def __init__(self, center, size):
        """
        Initialize a 3D box.
        :param center: Tuple (cx, cy, cz) for the box center.
        :param size: Tuple (width, height, depth) for the box dimensions.
        """
        self.cx, self.cy, self.cz = center
        self.w, self.h, self.d = size
        self.min_x = self.cx - self.w / 2
        self.max_x = self.cx + self.w / 2
        self.min_y = self.cy - self.h / 2
        self.max_y = self.cy + self.h / 2
        self.min_z = self.cz - self.d / 2
        self.max_z = self.cz + self.d / 2

    def density(self, x, y, z):
        if (self.min_x <= x <= self.max_x and
            self.min_y <= y <= self.max_y and
            self.min_z <= z <= self.max_z):
            return float('inf')
        else:
            return 0.0

class Sphere:
    def __init__(self, center, radius):
        """
        Initialize a sphere.
        :param center: Tuple (x, y, z) for the sphere center.
        :param radius: Radius of the sphere.
        """
        self.cx, self.cy, self.cz = center
        self.radius = radius

    def density(self, x, y, z):
        dx = x - self.cx
        dy = y - self.cy
        dz = z - self.cz
        distance_squared = dx*dx + dy*dy + dz*dz
        if distance_squared <= self.radius * self.radius:
            return float('inf')
        else:
            return 0.0

class World:
    def __init__(self):
        self.shapes = []

    def add_shape(self, shape):
        self.shapes.append(shape)

    def density(self, x, y, z):
        for shape in self.shapes:
            if shape.density(x, y, z) == float('inf'):
                return float('inf')
        return 0.0
