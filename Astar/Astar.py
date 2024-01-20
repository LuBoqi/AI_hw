import random
import time
from queue import Queue

import numpy as np
from matplotlib import pyplot as plt


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


def caculCost(node, isDijkstra=False):
    if isDijkstra:
        return node.h
    return node.g + node.h


def caculDistance(node1, node2):
    # cost = ((node2.x - node1.x) ** 2 + (node2.y - node1.x) ** 2) ** 0.5
    cost = (abs(node2.x - node1.x) + abs(node2.y - node1.y))
    return cost


def findNeighbor(node, pathMap):
    neighbors = []
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        x, y = node.x + dx, node.y + dy
        if 0 <= x < len(pathMap) and 0 <= y < len(pathMap[0]) and pathMap[x][y] != 255:
            neighbors.append(Node(x, y, pathMap[x][y] == 255))
    return neighbors


def Astar(pathMap, start, goal):
    begin = Node(start[0], start[1], pathMap[start[0]][start[1]] == 1)
    end = Node(goal[0], goal[1], pathMap[goal[0]][goal[1]] == 1)
    openList = []
    closeList = []
    openList.append(begin)
    while openList:
        node = openList.pop(0)
        if node == end:
            path = []
            node = node.parent
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
                if neighbor not in openList:
                    neighbor.parent = node
                    neighbor.g = node.g + 1
                    neighbor.h = caculDistance(neighbor, end)
                    pathMap[neighbor.x][neighbor.y] = int(neighbor.g + neighbor.h)*2
                    openList.append(neighbor)
                else:
                    if node.g + 1 < neighbor.g:
                        neighbor.parent = node
                        neighbor.g = node.g + 1
        openList = sorted(openList, key=caculCost, reverse=False)
    return None


def plot_astar(grid, start, goal, path):
    norm = plt.Normalize(np.min(grid), np.max(grid))
    plt.imshow(grid, cmap='viridis', origin='upper', norm=norm, interpolation=None)
    # plt.imshow([[1 if cell == 1 else 0 for cell in row] for row in grid], cmap='Greys', origin='upper')
    plt.imshow(np.ma.masked_where(grid == 255, grid), cmap='Reds', norm=norm, origin='upper', interpolation='none')

    # 绘制起点和终点
    plt.plot(start[1], start[0], 'ro', label='Start')  # 红色点表示起点
    plt.plot(goal[1], goal[0], 'go', label='Goal')  # 绿色点表示终点

    # 绘制路径
    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, marker='o', color='blue', label='Path')

    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    Map = [[0 if random.random() < 0.7 else 255 for _ in range(20)] for _ in range(20)]
    Map = [[255, 255, 255, 0, 255, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0, 255],
           [0, 255, 255, 255, 255, 255, 0, 0, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 0, 0],
           [0, 255, 0, 255, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0],
           [255, 0, 0, 255, 255, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 255, 0, 0],
           [0, 0, 255, 0, 255, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255],
           [255, 255, 255, 0, 0, 0, 255, 0, 255, 0, 0, 255, 0, 255, 255, 0, 0, 0, 255, 0],
           [0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 255, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0],
           [255, 0, 0, 255, 0, 0, 0, 0, 255, 0, 0, 0, 0, 255, 0, 255, 0, 0, 0, 255],
           [0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 255, 255, 0, 0, 0, 255, 0, 255, 0, 0],
           [255, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0],
           [0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 255, 255, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 255],
           [0, 255, 0, 0, 255, 0, 255, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0],
           [0, 0, 0, 0, 0, 0, 255, 0, 255, 0, 0, 255, 255, 0, 255, 0, 255, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 255, 0, 255, 0, 0],
           [255, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0],
           [0, 255, 255, 255, 0, 0, 255, 255, 0, 255, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0],
           [255, 0, 255, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0],
           [0, 255, 0, 0, 255, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255]]
    start = [int(random.random() * (len(Map) - 1)), int(random.random() * (len(Map[0]) - 1))]
    goal = [int(random.random() * (len(Map) - 1)), int(random.random() * (len(Map[0]) - 1))]
    start = [16, 5]
    goal = [2, 15]
    Map[start[0]][start[1]] = 0
    Map[goal[0]][goal[1]] = 0
    Aans = Astar(Map, start, goal)
    if Aans:
        print(start, 'to', goal)
        for line in Map:
            print(line)
        plot_astar(Map, start, goal, Aans)
    else:
        print('There is no path!')
