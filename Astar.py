from queue import Queue


class Node:
    def __init__(self, x, y, isHinder):
        self.x = x
        self.y = y
        self.isHinder = isHinder
        self.parent = None
        self.g = 0
        self.h = 0

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


def caculCost(node):
    return node.g + node.h


def caculDistance(node1, node2):
    cost = ((node2.x - node1.x) ** 2 + (node2.y - node1.x) ** 2) ** 0.5
    return cost


def findNeighbor(node, pathMap):
    neighbors = []
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        x, y = node.x + dx, node.y + dy
        if 0 <= x < len(pathMap) and 0 <= y < len(pathMap[0]) and not pathMap[x][y]:
            neighbors.append(Node(x, y, pathMap[x][y]))
    return neighbors


def Astar(pathMap, start, goal):
    begin = Node(start[0], start[1], pathMap[start[0]][start[1]])
    end = Node(goal[0], goal[1], pathMap[goal[0]][goal[1]])
    openList = Queue()
    closeList = []
    openList.put(begin)
    while not openList.empty():
        node = openList.get()
        if node == end:
            path = []
            while node.parent is not None:
                path.append((node.x, node.y))
                node = node.parent
            return path[::-1]

        closeList.append(node)
        neighbors = findNeighbor(node, pathMap)
        for neighbor in neighbors:
            if neighbor in closeList:
                continue
            if not neighbor.isHinder:
                if neighbor not in openList.queue:
                    neighbor.parent = node
                    neighbor.g = node.g + 1
                    neighbor.h = caculDistance(neighbor, end)
                    openList.put(neighbor)
                else:
                    if node.g + 1 < neighbor.g:
                        neighbor.parent = node
                        neighbor.g = node.g + 1
        sorted(openList.queue, key=caculCost, reverse=False)
    return None


def GreedyAlgorithm(pathMap, start, goal):
    pass


if __name__ == '__main__':
    Map = [[0, 1, 0, 0, 0],
           [0, 1, 0, 0, 0],
           [0, 1, 0, 0, 0],
           [0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0]
           ]
    Aans = Astar(Map, [0, 0], [2, 2])
    if Aans:
        print('Astar solution:')
        for (x, y) in Aans:
            Map[x][y] = '*'
        for line in Map:
            print(line)
    else:
        print('There is no path!')
