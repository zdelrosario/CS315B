import "regent"

-- Helper module to handle command line arguments
local PageRankConfig = require("pagerank_config")

local c = terralib.includecstring([[
#include "legion_c.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
 ]])

fspace Page {
  rank         : double;
  sum          : double; -- temp value for summation
  n_in         : long;   -- number of inward links
  n_out        : long;   -- number of outward links
}

fspace Link (p: region(Page)) {
  ptr_out : ptr(Page, p);
  ptr_in  : ptr(Page, p);
}

terra skip_header(f : &c.FILE)
  var x : uint64, y : uint64
  c.fscanf(f, "%llu\n%llu\n", &x, &y)
end

terra read_ids(f : &c.FILE, page_ids : &uint32)
  return c.fscanf(f, "%d %d\n", &page_ids[0], &page_ids[1]) == 2
end

task initialize_graph(r_pages   : region(Page),
                      r_links   : region(Link(r_pages)),
                      damp      : double,
                      num_pages : uint64,
                      filename  : int8[512])
where
  reads writes(r_pages, r_links)
do
  var ts_start = c.legion_get_current_time_in_micros()
  for page in r_pages do
    page.rank = 1.0 / num_pages
    page.sum = 0
  end

  var f = c.fopen(filename, "rb")
  skip_header(f)
  var page_ids : uint32[2]
  for link in r_links do
    regentlib.assert(read_ids(f, page_ids), "Less data that it should be")
    var src_page = unsafe_cast(ptr(Page, r_pages), page_ids[0])
    var dst_page = unsafe_cast(ptr(Page, r_pages), page_ids[1])
    link.ptr_out = src_page
    link.ptr_in  = dst_page
    src_page.n_out += 1
    dst_page.n_in  += 1
  end
  c.fclose(f)
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("Graph initialization took %.4f sec\n", (ts_stop - ts_start) * 1e-6)
end

--
-- TODO: Implement PageRank. You can use as many tasks as you want.
--

task pr_reduce(r_pages : region(Page),
	       r_links : region(Link(r_pages)))
where
  reads(r_links, r_pages.n_out, r_pages.rank), reduces +(r_pages.sum)
do
  -- Iterate over all links, reduce to sum
  for link in r_links do
    link.ptr_in.sum += link.ptr_out.rank / link.ptr_out.n_out
  end
end

task pr_rank(r_pages : region(Page),
	     damp    : double,
	     n_pages : long)
where
  reads writes(r_pages.rank, r_pages.sum)
do
  -- Iterate over all pages, compute new rank, reset
  var err  : double; err  = 0
  var prev : double; prev = 0
  for page in r_pages do
    -- Compute new rank
    prev = page.rank
    page.rank = (1-damp)/n_pages + damp*page.sum
    page.sum  = 0
    -- Add to error
    err += c.fabs(prev-page.rank)
  end
  return err
end

task pr_iter(r_pages : region(Page),
	     r_links : region(Link(r_pages)),
	     damp    : double,
	     n_pages : long)
where
  reads(r_pages, r_links), writes(r_pages)
do
  -- accumulate error for convergence check
  var err : double
  err = 0
  -- Iterate over links, accumulate the sums
  pr_reduce(r_pages, r_links)
  -- Iterate over nodes, compute new rank
  err += pr_rank(r_pages, damp, n_pages)
  -- Check error for convergence
  return err
end

task dump_ranks(r_pages  : region(Page),
                filename : int8[512])
where
  reads(r_pages.rank)
do
  var f = c.fopen(filename, "w")
  for page in r_pages do c.fprintf(f, "%g\n", page.rank) end
  c.fclose(f)
end

task toplevel()
  var config : PageRankConfig
  config:initialize_from_command()
  c.printf("**********************************\n")
  c.printf("* PageRank                       *\n")
  c.printf("*                                *\n")
  c.printf("* Number of Pages  : %11lu *\n",  config.num_pages)
  c.printf("* Number of Links  : %11lu *\n",  config.num_links)
  c.printf("* Damping Factor   : %11.4f *\n", config.damp)
  c.printf("* Error Bound      : %11g *\n",   config.error_bound)
  c.printf("* Max # Iterations : %11u *\n",   config.max_iterations)
  c.printf("* # Parallel Tasks : %11u *\n",   config.parallelism)
  c.printf("**********************************\n")

  -- Create a regions of pages and links
  var r_pages = region(ispace(ptr, config.num_pages), Page)
  var r_links = region(ispace(ptr, config.num_links), Link(wild))

  -- Allocate all the pages
  new(ptr(Link(r_pages), r_links), config.num_links)
  new(ptr(Page, r_pages), config.num_pages)

  --
  -- TODO: Create partitions for links and pages.
  --       You can use as many partitions as you want.
  --

  -- Initialize the page graph from a file
  initialize_graph(r_pages, r_links, config.damp, config.num_pages, config.input)

  var num_iterations = 0
  var converged = false
  var ts_start = c.legion_get_current_time_in_micros()
  var err : double

  while not converged do
    num_iterations += 1
    -- Perform iteration
    err = pr_iter(r_pages, r_links, config.damp, config.num_pages)
    -- Convergence check
    if err < config.error_bound then break end
    -- Iteration check
    if num_iterations >= config.max_iterations then
      c.printf("Maximum iterations reached!\n")
      break
    end
  end
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("PageRank converged after %d iterations in %.4f sec\n",
    num_iterations, (ts_stop - ts_start) * 1e-6)

  if config.dump_output then dump_ranks(r_pages, config.output) end
end

regentlib.start(toplevel)
