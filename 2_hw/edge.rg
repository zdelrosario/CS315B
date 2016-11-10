import "regent"

-- Helper modules to handle PNG files and command line arguments
local png        = require("png_util")
local EdgeConfig = require("edge_config")
local coloring   = require("coloring_util")

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

task factorize(parallelism : int) : int2d
  var limit = [int](cmath.sqrt([double](parallelism)))
  var size_x = 1
  var size_y = parallelism
  for i = 1, limit + 1 do
    if parallelism % i == 0 then
      size_x, size_y = i, parallelism / i
      if size_x > size_y then
        size_x, size_y = size_y, size_x
      end
    end
  end
  return int2d { size_x, size_y }
end

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
                filename : int8[256])
where
  reads writes(r_image)
do
  png.read_png_file(filename,
                    __physical(r_image.original),
                    __fields(r_image.original),
                    r_image.bounds)
  for e in r_image do
    r_image[e].smooth = r_image[e].original
    r_image[e].gradient = {0, 0}
    r_image[e].local_maximum = true
  end
  return 1
end

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

-- TODO: Copy and paste your 'sobelX', 'sobelY', and 'suppressNonmax' tasks here
task sobelX(r_image    : region(ispace(int2d), Pixel),
            r_interior : region(ispace(int2d), Pixel))
where
  reads(r_image.smooth), writes(r_interior.gradient.x)
do
  var ts_start = c.legion_get_current_time_in_micros()
  for e in r_interior do
     -- var grad_x : double = 0
     -- grad_x +=
     r_interior[e].gradient.x =
	-1. * r_image[e + {-1,-1}].smooth +
	-2. * r_image[e + { 0,-1}].smooth +
	-1. * r_image[e + { 1,-1}].smooth +
	 1. * r_image[e + {-1, 1}].smooth +
	 2. * r_image[e + { 0, 1}].smooth +
	 1. * r_image[e + { 1, 1}].smooth
     -- r_interior[e].gradient.x = grad_x
  end
  var ts_end = c.legion_get_current_time_in_micros()
  c.printf("Sobel operator on x-axis took %.3f sec.\n", (ts_end - ts_start) * 1e-6)
end

task sobelY(r_image    : region(ispace(int2d), Pixel),
            r_interior : region(ispace(int2d), Pixel))
where
  reads(r_image.smooth), writes(r_interior.gradient.y)
do
  var ts_start = c.legion_get_current_time_in_micros()
  for e in r_interior do
     -- var grad_y : double = 0
     -- grad_y +=
     r_interior[e].gradient.y =
	-1. * r_image[e + {-1,-1}].smooth +
	-2. * r_image[e + {-1, 0}].smooth +
	-1. * r_image[e + {-1, 1}].smooth +
	 1. * r_image[e + { 1,-1}].smooth +
	 2. * r_image[e + { 1, 0}].smooth +
	 1. * r_image[e + { 1, 1}].smooth
     -- r_interior[e].gradient.y = grad_y
  end
  var ts_end = c.legion_get_current_time_in_micros()
  c.printf("Sobel operator on y-axis took %.3f sec.\n", (ts_end - ts_start) * 1e-6)
end

task suppressNonmax(r_image    : region(ispace(int2d), Pixel),
                    r_interior : region(ispace(int2d), Pixel))
where
  -- reads(r_image.gradient), writes(r_interior.local_maximum)
  reads(r_image.gradient), writes(r_interior.local_maximum)
do
  var ts_start = c.legion_get_current_time_in_micros()
  for e in r_interior do
     -- Compute rounded angle
     var angle : int = 0
     angle = cmath.atan(r_image[e].gradient.y/r_image[e].gradient.x)*180./PI
     angle = cmath.floor(angle/45) % 4
     --
     if angle == 0 then
       -- Case 0
       if r_image[e].gradient:norm()<r_image[e+{0, 1}].gradient:norm() or
          r_image[e].gradient:norm()<r_image[e+{0,-1}].gradient:norm() then
         e.local_maximum = false
       end
     elseif angle == 1 then
       -- Case 1
       if r_image[e].gradient:norm()<r_image[e+{ 1, 1}].gradient:norm() or
          r_image[e].gradient:norm()<r_image[e+{-1,-1}].gradient:norm() then
         e.local_maximum = false
       end
     elseif angle == 2 then
       -- Case 2
       if r_image[e].gradient:norm()<r_image[e+{ 1,0}].gradient:norm() or
          r_image[e].gradient:norm()<r_image[e+{-1,0}].gradient:norm() then
         e.local_maximum = false
       end
     elseif angle == 3 then
       -- Case 3
       if r_image[e].gradient:norm()<r_image[e+{ 1,-1}].gradient:norm() or
          r_image[e].gradient:norm()<r_image[e+{-1, 1}].gradient:norm() then
         e.local_maximum = false
       end
     end
  end
  var ts_end = c.legion_get_current_time_in_micros()
  c.printf("Non-maximum suppression took %.3f sec.\n", (ts_end - ts_start) * 1e-6)
end
--

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

task saveEdge(r_image : region(ispace(int2d), Pixel),
              filename : int8[256])
where
  reads(r_image.edge)
do
  png.write_png_file(filename,
                     __physical(r_image.edge),
                     __fields(r_image.edge),
                     r_image.bounds)
end

task block_task(r_image : region(ispace(int2d), Pixel))
where
  reads writes(r_image)
do
  return 1
end

terra wait_for(x : int) return 1 end

task toplevel()
  var config : EdgeConfig
  config:initialize_from_command()

  -- Create a logical region for original image and intermediate results
  var size_image = png.get_image_size(config.filename_image)
  var r_image = region(ispace(int2d, size_image), Pixel)

  -- Create a sub-region for the interior part of image
  var p_interior = create_interior_partition(r_image)
  var r_interior = p_interior[0]

  -- Create an equal partition of the interior image
  var p_private_colors = ispace(int2d, factorize(config.parallelism))
  var p_private = partition(equal, r_interior, p_private_colors)

  -- Create a halo partition for ghost access
  var c_halo = coloring.create()
  -- var c_halo = c.legion_domain_coloring_create()
  for color in p_private_colors do
    var bounds = p_private[color].bounds
    var halo_bounds : rect2d = {bounds.lo-{2,2},bounds.hi+{2,2}}
    coloring.color_domain(c_halo, color, halo_bounds)
  end
  --
  -- TODO: Create an aliased partition of region 'r_image'
  --       using coloring 'c_halo':
  -- var p_halo = partition(...)
  var p_halo = partition(aliased,          -- Type of disjointness
			 r_image,          -- Region to partition
			 c_halo,           -- Coloring object
			 p_private_colors) -- Colors
  --
  coloring.destroy(c_halo)

  var token = initialize(r_image, config.filename_image)
  wait_for(token)
  var ts_start = c.legion_get_current_time_in_micros()

  --
  -- TODO: Change the following task launches so they are launched for
  --       each of the private regions and its halo region.
  --
  for color in p_private.colors do
    smooth(p_halo[color], p_private[color])
  end

  for color in p_private.colors do
    sobelX(p_halo[color], p_private[color])
    sobelY(p_halo[color], p_private[color])
  end

  for color in p_private.colors do
    suppressNonmax(p_halo[color], p_private[color])
   end

  --
  -- Launch task 'edgefromGradient' for each of the private regions.
  -- This will be optimized to a parallel task launch.
  --
  for color in p_private.colors do
    edgeFromGradient(p_private[color], config.threshold)
  end

  for color in p_private_colors do
    token += block_task(p_private[color])
  end
  wait_for(token)
  var ts_end = c.legion_get_current_time_in_micros()
  c.printf("Total time: %.6f sec.\n", (ts_end - ts_start) * 1e-6)

  saveEdge(r_image, config.filename_edge)
end

regentlib.start(toplevel)
