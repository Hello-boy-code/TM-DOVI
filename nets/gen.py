def gilbert2d(width, height):
    if width >= height:
        yield from generate2d(0, 0, width, 0, 0, height)
    else:
        yield from generate2d(0, 0, 0, height, width, 0)


def generate2d(x, y, ax, ay, bx, by):
    w = abs(ax + ay)
    h = abs(bx + by)
    dax, day = (1 if ax + ay > 0 else -1, 0) if ax != 0 else (0, 1 if ay > 0 else -1)
    dbx, dby = (1 if bx + by > 0 else -1, 0) if bx != 0 else (0, 1 if by > 0 else -1)

    if h == 1:
        for _ in range(w):
            yield (x, y)
            x += dax
            y += day
        return
    if w == 1:
        for _ in range(h):
            yield (x, y)
            x += dbx
            y += dby
        return

    ax2, ay2 = ax // 2, ay // 2
    bx2, by2 = bx // 2, by // 2

    if 2 * w > 3 * h:
        if (ax2 + ay2) % 2 != 0 and w > 2:
            ax2 += dax
            ay2 += day
        yield from generate2d(x, y, ax2, ay2, bx, by)
        yield from generate2d(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by)
    else:
        if (bx2 + by2) % 2 != 0 and h > 2:
            bx2 += dbx
            by2 += dby
        yield from generate2d(x, y, bx2, by2, ax2, ay2)
        yield from generate2d(x + bx2, y + by2, ax, ay, bx - bx2, by - by2)
        yield from generate2d(
            x + (ax - dax) + (bx2 - dbx),
            y + (ay - day) + (by2 - dby),
            -bx2, -by2, -(ax - ax2), -(ay - ay2)
        )