"""One terminal line per command, with streaming counts and replaceable phases."""

from tqdm import tqdm


class Progress:
    def __init__(self, description, *, total=None, initial=0, unit="image"):
        self.bar = None
        self.description = description
        self.total = total
        self.initial = initial
        self.unit = unit

    def __enter__(self):
        self.phase(self.description, total=self.total, initial=self.initial, unit=self.unit)
        return self

    def __exit__(self, *exc):
        self.bar.close()

    def phase(self, description, *, total=None, initial=0, unit="image", **counts):
        # Clear the previous phase without leaving a line. A fresh timer excludes
        # earlier phases and previously completed items from the new phase's ETA.
        if self.bar is not None:
            self.bar.leave = False
            self.bar.close()
        self.bar = tqdm(
            desc=description,
            total=total,
            initial=initial,
            unit=unit,
            postfix=counts or None,
            disable=None,
            dynamic_ncols=True,
            mininterval=0.2,
        )

    def status(self, description):
        self.bar.set_description_str(description)

    def update(self, n=1, **counts):
        if counts:
            self.bar.set_postfix(counts, refresh=False)
        self.bar.update(n)

    def track(self, iterable, *, stats=None):
        """Count an item after its body finishes, including a normal continue."""
        for item in iterable:
            yield item
            self.update(**(stats() if stats else {}))
