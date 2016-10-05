import "regent"

-- Helper modules to handle PNG files and command line arguments
local png        = require("png_util")
local EdgeConfig = require("edge_config")

-- Some C APIs
local c     = regentlib.c
local sqrt  = regentlib.sqrt(double)
local cmath = terralib.includec("math.h")
local PI = cmath.M_PI

-- 2D vector type
struct Vector2d
{
  x : double;
  y : double;
}

terra Vector2d:norm()
  return sqrt(self.x * self.x + self.y * self.y)
end

terra Vector2d.metamethods.__div(v : Vector2d, c : double)
  return Vector2d { v.x / c, v.y / c }
end

-- Field space for pixels
fspace Pixel
{
  original      : uint8;    -- Original pixel in 8-bit gray scale
  smooth        : uint8;    -- Pixel after Gaussian smoothing
  gradient      : Vector2d; -- Gradient vector
  local_maximum : bool;     -- Marks if the gradient is a local maximum
  edge          : uint8;    -- Extracted edge
}

task create_interior_partition(r_image : region(ispace(int2d), Pixel))
  var coloring = c.legion_domain_coloring_create()
  var bounds = r_image.ispace.bounds
  c.legion_domain_coloring_color_domain(coloring, 0,
    rect2d { bounds.lo + {2, 2}, bounds.hi - {2, 2} })
  var interior_image_partition = partition(disjoint, r_image, coloring)
  c.legion_domain_coloring_destroy(coloring)
  return interior_image_partition
end

--
-- The 'initialize' task reads the image data from the file and initializes
-- the fields for later tasks. The upper left and lower right corners of the image
-- correspond to point {0, 0} and {width - 1, height - 1}, respectively.
--
task initialize(r_image : region(ispace(int2d), Pixel),
                filename : rawstring)
where
  reads writes(r_image)
do
  png.read_png_file(filename,
                    __physical(r_image.original),
                    __fields(r_image.original),
                    r_image.bounds)
  copy(r_image.original, r_image.smooth)
  fill(r_image.gradient, {0, 0})
  fill(r_image.local_maximum, true)
end

--
-- The 'smooth' task implements Gaussian smoothing, which is a convolution
-- between the image and the following 5x5 filter:
--
--        |  2  4  5  4  2 |
--   1    |  4  9 12  9  4 |
--  --- * |  5 12 15 12  5 |
--  159   |  4  9 12  9  4 |
--        |  2  4  5  4  2 |
--
-- Note that the upper left corner of the filter is applied to the
-- pixel that is off from the center by (-2, -2).
--
task smooth(r_image    : region(ispace(int2d), Pixel),
            r_interior : region(ispace(int2d), Pixel))
where
  reads(r_image.original), writes(r_interior.smooth)
do
  var ts_start = c.legion_get_current_time_in_micros()
  for e in r_interior do
    var smooth : double = 15 * r_image[e].original
    for polarity = 1, -2, -2 do
      smooth +=
        12.0 * r_image[e + {1 * polarity, 0 * polarity}].original +
         9.0 * r_image[e + {1 * polarity, 1 * polarity}].original +
         4.0 * r_image[e + {1 * polarity, 2 * polarity}].original +
         5.0 * r_image[e + {2 * polarity, 0 * polarity}].original +
         4.0 * r_image[e + {2 * polarity, 1 * polarity}].original +
         2.0 * r_image[e + {2 * polarity, 2 * polarity}].original

      smooth +=
        12.0 * r_image[e + {0 * polarity, 1 * -polarity}].original +
         9.0 * r_image[e + {1 * polarity, 1 * -polarity}].original +
         4.0 * r_image[e + {2 * polarity, 1 * -polarity}].original +
         5.0 * r_image[e + {0 * polarity, 2 * -polarity}].original +
         4.0 * r_image[e + {1 * polarity, 2 * -polarity}].original +
         2.0 * r_image[e + {2 * polarity, 2 * -polarity}].original
    end
    r_interior[e].smooth = [uint8](smooth / 159.0)
  end
  var ts_end = c.legion_get_current_time_in_micros()
  c.printf("Gaussian smoothing took %.3f sec.\n", (ts_end - ts_start) * 1e-6)
end

task saveSmooth(r_image : region(ispace(int2d), Pixel),
                filename : rawstring)
where
  reads(r_image.smooth)
do
  png.write_png_file(filename,
                     __physical(r_image.smooth),
                     __fields(r_image.smooth),
                     r_image.bounds)
end

--
-- TODO: Implement task 'sobelX'
--
-- The 'sobelX' task finds x component of the gradient vector at each pixel.
-- Use the following 3x3 filter to implement this task:
--
--  | -1  0  1 |
--  | -2  0  2 |
--  | -1  0  1 |
--
task sobelX(r_image    : region(ispace(int2d), Pixel),
            r_interior : region(ispace(int2d), Pixel))
-- TODO: Provide necessary privileges for this task
--where
--  reads(...), writes(...)
--do
  var ts_start = c.legion_get_current_time_in_micros()
  for e in r_interior do
    -- TODO: Fill the body of this loop
  end
  var ts_end = c.legion_get_current_time_in_micros()
  c.printf("Sobel operator on x-axis took %.3f sec.\n", (ts_end - ts_start) * 1e-6)
end

--
-- TODO: Implement task 'sobelY'
--
-- The 'sobelY' task finds y component of the gradient vector at each pixel.
-- Use the following 3x3 filter to implement this task:
--
--  | -1 -2 -1 |
--  |  0  0  0 |
--  |  1  2  1 |
--
task sobelY(r_image    : region(ispace(int2d), Pixel),
            r_interior : region(ispace(int2d), Pixel))
-- TODO: Provide necessary privileges for this task
--where
--  reads(...), writes(...)
--do
  var ts_start = c.legion_get_current_time_in_micros()
  for e in r_interior do
    -- TODO: Fill the body of this loop
  end
  var ts_end = c.legion_get_current_time_in_micros()
  c.printf("Sobel operator on y-axis took %.3f sec.\n", (ts_end - ts_start) * 1e-6)
end

task edgeFromGradient(r_image : region(ispace(int2d), Pixel),
                      threshold : double)
where
  reads(r_image.{gradient, local_maximum}),
  writes(r_image.edge)
do
  for e in r_image do
    if e.local_maximum and e.gradient:norm() >= threshold then
      e.edge = 255
    end
  end
end

--
-- TODO: Implement task 'suppressNonmax'
--
-- The 'suppressNonmax' task filters only the gradients that are local maximum.
-- Each gradient is compared with two neighbors along its positive
-- and negative direction. The gradient direction is rounded to nearest 45°
-- to work on a discrete image. The following diagram will be useful to
-- determine which neighbors to pick for the comparison:
--
--           j - 1    j      j + 1    x-axis
--         -------------------------
--         |  45°  |  90°  |  135° |
--  i - 1  |  or   |  or   |  or   |
--         |  225° |  270° |  315° |
--         |------------------------
--         |  0°   |       |  0°   |
--    i    |  or   | center|  or   |
--         |  180° |       |  180° |
--         |------------------------
--         |  135° |  90°  |  45°  |
--  i + 1  |  or   |  or   |  or   |
--         |  315° |  270° |  225° |
--         -------------------------
--  y-axis
--
-- Hint: You might want to call some math functions with module 'cmath' imported
--       above.
--
task suppressNonmax(r_image    : region(ispace(int2d), Pixel),
                    r_interior : region(ispace(int2d), Pixel))
-- TODO: Provide necessary privileges for this task
--where
--  reads(...), writes(...)
--do
  var ts_start = c.legion_get_current_time_in_micros()
  for e in r_interior do
    -- TODO: Fill the body of this loop
  end
  var ts_end = c.legion_get_current_time_in_micros()
  c.printf("Non-maximum suppression took %.3f sec.\n", (ts_end - ts_start) * 1e-6)
end

task saveEdge(r_image : region(ispace(int2d), Pixel),
              filename : rawstring)
where
  reads(r_image.edge)
do
  png.write_png_file(filename,
                     __physical(r_image.edge),
                     __fields(r_image.edge),
                     r_image.bounds)
end

task toplevel()
  var config : EdgeConfig
  config:initialize_from_command()

  -- Create a logical region for original image and intermediate results
  var size_image = png.get_image_size(config.filename_image)
  var r_image = region(ispace(int2d, size_image), Pixel)

  -- Create a sub-region for the interior part of image
  var p_interior = create_interior_partition(r_image)
  var r_interior = p_interior[0]

  initialize(r_image, config.filename_image)

  if not config.skip_smooth then
    smooth(r_image, r_interior)
    if config.save_smooth then
      saveSmooth(r_image, config.filename_smooth)
    end
  end

  sobelX(r_image, r_interior)
  sobelY(r_image, r_interior)

  if not config.skip_suppress then
    suppressNonmax(r_image, r_interior)
  end

  edgeFromGradient(r_interior, config.threshold)
  saveEdge(r_image, config.filename_edge)
end

regentlib.start(toplevel)
