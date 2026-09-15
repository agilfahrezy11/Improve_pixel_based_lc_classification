"""Apply a fitted classifier to a raster and write hard class predictions."""

from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window


PathLikeInput = str | PathLike[str]


def classify_raster(
	raster_path: PathLikeInput,
	model: Any,
	output_path: PathLikeInput,
	*,
	selector: Any | None = None,
	block_size: int = 512,
	nodata: int | float = 0,
	output_dtype: str | np.dtype[Any] | None = None,
) -> Path:
	"""Classify every valid raster pixel with a fitted model.

	The raster bands are passed to ``model.predict`` in their original order.
	When ``selector`` is supplied, it must be the fitted selector used during
	training. Invalid or non-finite pixels receive ``nodata``. The output is a
	single-band GeoTIFF that preserves the source raster's spatial metadata.
	"""
	if block_size < 1:
		raise ValueError("block_size must be at least 1")

	output_path = Path(output_path)
	with rasterio.open(raster_path) as source:
		if output_dtype is None:
			if not hasattr(model, "classes_"):
				raise ValueError(
					"output_dtype is required when the model has no classes_ attribute"
				)
			output_dtype = np.asarray(model.classes_).dtype
		output_dtype = np.dtype(output_dtype)
		if output_dtype.kind not in "biuf":
			raise ValueError("output_dtype must be a numeric raster data type")

		profile = source.profile.copy()
		profile.update(
			count=1,
			dtype=output_dtype.name,
			nodata=nodata,
			compress="deflate",
		)

		with rasterio.open(output_path, "w", **profile) as destination:
			for row_start in range(0, source.height, block_size):
				height = min(block_size, source.height - row_start)
				for column_start in range(0, source.width, block_size):
					width = min(block_size, source.width - column_start)
					window = Window(column_start, row_start, width, height)
					block = source.read(window=window, masked=True)

					pixels = np.ma.getdata(block).reshape(source.count, -1).T
					masked = np.ma.getmaskarray(block).reshape(source.count, -1).T
					valid = ~masked.any(axis=1)
					valid &= np.isfinite(pixels).all(axis=1)

					classified = np.full(
						(height * width), nodata, dtype=output_dtype
					)
					if valid.any():
						features = pixels[valid]
						if selector is not None:
							features = selector.transform(features)
							if hasattr(features, "toarray"):
								features = features.toarray()
						classified[valid] = np.asarray(
							model.predict(features), dtype=output_dtype
						)

					destination.write(
						classified.reshape(height, width), 1, window=window
					)

	return output_path
