### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ a9710e38-25ef-11ec-2bea-c152dd5b1b5c
begin
	using DataFrames
	using CSV
	using Shapefile
	using Plots
	using PolygonOps
	using LaTeXStrings
	using Distributions
	using StatsPlots
end

# ╔═╡ af7df31b-dafe-4ac7-9a26-958a7c2b3ebe
md"""
# Overview
To begin my DS6040 Bayesian ML project, I want to do a bit of exploratory analysis on the data I inted to work with for the project. The data consists of water pump functionality information for Tanzania and is hosted by [DataDriven](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/). The competition at the provided URL focuses on classifier accuracy -- namely developing a classifier that can accurately predict whether a particular water pump is functioning or not. I think this is an interesting question, but it will not be the primary focus of my project. What I intend to do is develop a Bayesian classifier that allows us to describe our uncertainty about whether a water pump is working and then use those posterior distributions to develop an optimal repair strategy.

To start this effort, as mentioned above, I want to first get a good understanding of the data. In particular, I think geography will be an interesting component of this analysis and it's something I want to first explore. Thus the purpose of this first notebook is just to get my hands on the data and start manipulating before I think about developing a Bayesian model.
"""

# ╔═╡ a43cb588-ddd9-4cf2-a9cd-1a3c1e9d4005
Xfile = "/home/zblanks/Documents/uva/fall21/bayesian-ml/project/data/X.csv";

# ╔═╡ 0b8f5add-00d2-49ec-b598-6ff34670d195
yfile = "/home/zblanks/Documents/uva/fall21/bayesian-ml/project/data/y.csv";

# ╔═╡ 040f381a-0cda-40d4-bec6-4f4a813adf5d
shp_file = "/home/zblanks/Documents/uva/fall21/bayesian-ml/project/data/tza/tza.shp";

# ╔═╡ 0d59f5e4-de99-4aaf-9cd5-d47dfa3fd9b4
df = DataFrame(CSV.File(Xfile));

# ╔═╡ 256e435f-3471-4eb1-9d59-3db531168b42
md"""
To start with, let's just print out all the columns and see what we have. A full description of the data is provided at [DataDriven](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/25/)

Additionally, one can find the Shapefile that I'm using throughout this notebook at [OCHA](https://data.humdata.org/dataset/tanzania-administrative-boundaries-level-1-to-3-regions-districts-and-wards-with-2012-population)
"""

# ╔═╡ 00b066f4-1224-4b6f-a335-a3d1da1123f4
names(df)

# ╔═╡ 1e73a7c1-88cd-4766-b314-cdd5ca09a9f2
md"""
Some of these variables will likely be interesting later, but as I mentioned above I want to first focus on the geographic dispersion of the water pumps throughout Tanzania. 

One of the columns in the DataFrame is titled `Region`. According to my brief Wikipedia search, Tanzania is partitioned into various administrative regions. For example, the [Iringa Region](https://en.wikipedia.org/wiki/Iringa_Region), is one of the region that show in the data and is also one of the $31$ administrative region for Tanzania. Fortunately, we're also provided the latitude and longitude coordinates for each of the water pumps, so I should be able to plot their locations (assuming Julia has this capability which would be quite surprising if it didn't).
"""

# ╔═╡ 169be54b-4562-4b1c-8cae-a27c07b054d1
md"""
## Exploring Tanzania's Geography
"""

# ╔═╡ 757b5ce2-63b8-4266-a00a-3b545b01c8ed
table = Shapefile.Table(shp_file);

# ╔═╡ ebfda391-25a7-41f1-b57f-964058c6d29c
shp = Shapefile.shapes(table);

# ╔═╡ 05dc6598-a645-45d0-a44f-d06229847d55
plot(shp)

# ╔═╡ ad9337fa-e7da-43eb-ad02-ca16bd520f47
md"""
The map I'm showing is the $31$ administrative regions of Tanzania. The parts that do not have any fill correspond to lakes or bodies of water and the $x$-axis corresponds to longitude and the $y$-axis denotes the lines of latitude. Now I want to show the geographic dispersal of water pumps throughout TZA.
"""

# ╔═╡ 95a34546-90b2-4df3-8c26-cb2e1d8d3022
begin
	p = plot(shp)
	scatter!(p, df.longitude, df.latitude, markersize=0.05, color="black")
end

# ╔═╡ 190f5259-61aa-4c04-8661-98ec9ef919ef
md"""
Interestingly, it appears that the data creators listed water pumps with (possibly) unknown coordinates at (longitude, latitude) = (0, 0). This clearly not possible because that would correspond to I believe somewhere in the middle of the ocean, so I'll need to clear out those entries for now. When I'm building my model I'll see if I can use some inferential logic to deduce approximately where those pumps may be located.
"""

# ╔═╡ 459e7e20-c65b-4d92-a45c-21d4e611df6f
df1 = df[(df.longitude .!= 0), :];

# ╔═╡ fb929142-d177-4685-b659-cb90b17cd605
begin
	p1 = plot(shp)
	scatter!(p1, df1.longitude, df1.latitude, markersize=0.1, color="black")
end

# ╔═╡ 3bd619a7-7d54-47e0-a8df-ca117ba9ad35
md"""
Unsurprisingly, water pumps appear to be concentrated in regions close to bodies of water. For example, the norther portion of Tanzania borders Lake Victoria -- one of the largest bodies of fresh water in the world (I believe). What's more interesting to me is areas where there are gaps in water pumps. This could be indicative of locations where there are fewer people in Tanzania or it could be that the data has simply not been recorded. I will have to do some basic research to understand population patterns in the country to have a greater intuition about this map.

First though, I want to briefly get a handle on the various administrative regions in TZA to see how they're refelected in our data.
"""

# ╔═╡ bd1e2ecf-dd98-4814-b161-190141b497d1
sort(unique(df1.region))

# ╔═╡ cb4f4a27-8603-4994-9547-9cfa43f81123
sort(unique(df1.region_code))

# ╔═╡ 7e40284e-18d4-4009-8569-36ae24cec7cc
md"""
This is a side-bar, but it does relate to the overall data exploration, there seems to be different specifications for region code. In this map: ![](https://mk0mapprpkwwgngeq1o.kinstacdn.com/wp-content/uploads/2021/01/image-428.jpeg)

The region is given its number alphebetically whereas our data seems to correspond with the standard provided in the Shapefile. Morever the polygons are ordered alphabetically (e.g., Arusha is first and thus it is the first entry the object `shp`.

I don't really case what standard we use for the data, but we do need to just pick one and be consistent. I think it'll be easier to just go with the alphabetical ordering because that's how the Shapefiles are arranged and that makes more sense than whatever ordering is provided in the data.

Other notes:

1. There are only $21$ unique regions in the data; I need to investigate why we're missing ten

2. Some of the unique values for region codes are for numbers that aren't possible (e.g, $99$). I also need to investigate where those belong.
"""

# ╔═╡ 5aac270a-3935-4338-9b1b-2fb1fdcef477
md"""
### Where are the missing regions?
"""

# ╔═╡ db7e2740-2598-4eb7-a226-8188989b4562
regions = table.ADM1_EN

# ╔═╡ c576974a-1e1f-4dc0-bfe2-5ed25075b607
sort(unique(df1.region))

# ╔═╡ 63cef946-7932-433a-9b99-635cf80f5ab0
setdiff(lowercase.(regions), lowercase.(unique(df1.region)))

# ╔═╡ 77e9f470-1df9-481e-b1fa-063508fbd7b9
md"""
Some of these missing regions don't seem correct. Just looking at the water pump location map, I can see a number of black dots in:

- Geita
- Njombe
- Simiyu
- Songwe
- Katavi

It's also not representing the islands in this data. That's less surprising to me; I imagine they're hard to record data, but I may to explore that hypothesis in greater detail.

Dar-es-Salaam is also in the data, there's just diagreement about one source using a dash and the other does not.

Apparently the region coding in the data is not fully accurate though, so that's something I'm going to have to fix especially since I think there are a number of proxy variables that I can use such as the wealth level of the particular administrative region.
"""

# ╔═╡ 9669a2cf-1417-4bc4-808d-858de9371d07
md"""
### Re-Mapping Points to Correct Administrative Region

To accomplish this task, I'll investigate the package: [PolygonOps.jl](https://github.com/JuliaGeometry/PolygonOps.jl). They have a function `inpolygon` which seems to implement a $2018$ paper titled *Optimal Reliable Point-in-Polygon Test and Differential Coding Boolean Operations on Polygons*. We'll see if it works for our case.
"""

# ╔═╡ 906f44f9-403d-411d-943e-d873795887fe
idx = findall(x -> x == "Arusha", df1.region)[1]

# ╔═╡ dab8667a-68b4-4bd4-96f6-dab0429cef73
lon, lat = df1[idx, ["longitude", "latitude"]]

# ╔═╡ 0e4f8f63-a2d5-438c-a580-e4a023467f4b
arusha = copy(shp[1].points)

# ╔═╡ a5c07da4-13a6-4a71-a187-aea28ea74add
push!(arusha, shp[1].points[1])

# ╔═╡ 85305f5f-177f-4b2c-9c84-cf1708eb6779
begin
	P = []
	for i = 1:length(arusha)
		push!(P, (arusha[i].x, arusha[i].y))
	end
	push!(P, (arusha[1].x, arusha[1].y))
end

# ╔═╡ 7587cc2a-64c7-40ab-af26-237eab25f41c
inpolygon((lon, lat), P)

# ╔═╡ 212d38a9-1d7c-4661-9c3f-defb707a86b6
md"""
So (assuming that the sample I selected is actually in Arusha) the `inpolygon` algorithm appears to work. To make it work I had to add the final point of the polygon to the end per the documentation at `PolygonOps.jl`. To generalize this approach to update the region value for each of the samples in the data, I need to have an algorithm that does approximately the following:

* Generates a polygon dictionary that maps each of the regions with their associated shapefile

* For each sample and each polygon, check if a particular (longitude, latitude) pair belongs to a particular region.
  * A good heuristic to improve computational performance would be to assume that the particular sample is correct and then check that. If it does belong to the polygon then we're good and don't have to check the other $30$ regions; otherwise, we'll have to do an exhaustive search.
"""

# ╔═╡ 0ffc9737-879a-48eb-8345-edaba4fb83f3
function create_polygon_dict(shp, regions)
	d = Dict()
	
	for (i, region) in enumerate(regions)
		P = []
		pts = shp[i].points
		for j = 1:length(pts)
			push!(P, (pts[j].x, pts[j].y))
		end
		
		# The PolygonOps.jl package requires the first & last entry to be the same
		push!(P, (pts[1].x, pts[1].y))
		d[region] = P
	end
	
	return d
end

# ╔═╡ 4305c38e-4b50-49b8-97b7-66e7265279d6
function check_region(df, shp, regions)
	n = size(df, 1)
	d = create_polygon_dict(shp, regions)
	new_regions = Vector{String}(undef, n)
	
	for i = 1:n
		lon, lat = df[i, ["longitude", "latitude"]]
		curr_region = df[i, "region"]
		
		# Do a simple check if the proposed region is correct; o.w. have to do 
		# an exhaustive search
		val = inpolygon((lon, lat), d[curr_region])
		if val == 1
			new_regions[i] = curr_region
		else
			for region in setdiff(regions, [curr_region])
				val = inpolygon((lon, lat), d[region])
				if val == 1
					new_regions[i] = region
				end
			end
		end
	end
	
	return new_regions
end

# ╔═╡ 5f2428c8-3a0b-4001-910a-94ac3465fc16
md"""
Now that I've define the two key functions to re-map the regions according to the Shapefiles, I just have to make sure the `regions` vector agrees with how its spelled in the data to ensure that works. Once I've done that computation we can then see if there are in fact water pumps not present in particular administrative regions.
"""

# ╔═╡ 26a65919-7896-40b8-8fb1-813f409f406f
df1.region = lowercase.(df1.region)

# ╔═╡ d5f34bc0-b1ee-402b-8033-04bfdefa1980
sort(unique(df1.region))

# ╔═╡ b088c955-8587-4128-b045-9c72c92acc9c
regions

# ╔═╡ 2d3ed9dd-c97a-43cc-b572-64f7979bc0c1
regions[2] = replace.(regions[2], "-" => " ")

# ╔═╡ 65c7a621-045d-4996-ad67-36dea8765167
test_regions = check_region(df1[1:100, :], shp, lowercase.(regions))

# ╔═╡ 67bc4d3d-e660-4a7b-bd70-616e38774be8
df1.region[1]

# ╔═╡ 30c130da-0c1d-476a-9c18-f8b509c4db9e
md"""
This sequence of computations is trivially parallelizable, but I don't want to kill my computer by doing it over the ~$57$k samples in the data. I'll see if I can get a Rivanna slot so that I can expedite this data transformation (along with others that I'm sure will come up). 

One item of concern though is that I see an `#undef` in the above vector. That suggests for the particular sample that the (lon, lat) coordinate did not belong to any of the polygons. This could mean it is on the border or that sample doesn't belong to TZA in general. What I'll do is expand the function to account for samples being on the border of the polygon
"""

# ╔═╡ d3183657-d469-4257-a14a-b3584a16c18f
test_regions[end-7]

# ╔═╡ 7a733e15-8531-4213-aea8-220664a3bfb9
df1[93, ["region", "longitude", "latitude"]]

# ╔═╡ 79956456-d383-48ba-8083-6fe8195af521
polygon_dict = create_polygon_dict(shp, lowercase.(regions))

# ╔═╡ d9c7ddbf-5e78-4430-9d2d-cf283536fd47
inpolygon((df1[93, "longitude"], df1[93, "latitude"]), polygon_dict["kagera"])

# ╔═╡ d8afca88-c70e-469f-b166-7e59eb635ae1
begin
	p3 = plot(shp[6])
	scatter!(p3, [df1[93, "longitude"]], [df1[93, "latitude"]])
end

# ╔═╡ 0654a284-ac3a-4c96-8814-5d4496cca986
md"""
That's very strange -- clearly the point is in Kagera, but the `inpolygon` algorithm is saying it's not in the polygon. I'm not sure what to do about that.

It may be prudent to investigate alternative packages because it seems that PolygonOps.jl is breaking for multiple cases in just the first $100$ samples. Additionally, it's likely a reasonable belief that most of the region samples are correct which raises the question (given how computationally intensive this approach is) of whether there's a smarter way to go about this problem.
"""

# ╔═╡ e8d154a1-0df4-438e-b724-af1773c89f0f
md"""
### Regional Re-Mapping Conclusion
After doing more research it seems that the data is using a different partitioning of administrative regions than the map I showed previously in this notebook. Moreover it seems like there are multiple standards in TZA and there doesn't seem to be universal agreement on how to do so. My working hypothesis is that the Tazanian government changed the administrative regions at some point in the 2010s and this is reflected in the different data standards. So here's what I am going to do:

- We're going to use the region that is provided in the data. The marginal benefit of re-mapping to the $31$ administrative regions using a polygon search algorithm is just not worth it. I imagine the socio-economic differences between a merged region is not that wildly different. This is also compounded by the fact that there seems to be some bugs with the `PolygonOps.jl` package (see the Kagera example above).

- The (longitude, latitude) coordinates will be more useful anyways. My current thoughts are:
  - There is probbaly a relationship between region and how many working water pumps there are. This could be a function of GDP but it could also be a function of their proximity to water sources (e.g., lakes, etc.)
  - I suspect there may also be a relationship between locations that are far away from villages and the proportion of working pumps
"""

# ╔═╡ d58f6eed-8002-4363-820f-b959ff19fa8e
md"""
## Regions, Villages, & Water Pump Functionality

After that wonderful escapade into the intricacies of Tanzanian administrative governance, I now want to focus on seeing if there's any sort of relationship between the proportion of functioning water pumps in a given administrative region and its GDP. To start with, let's just investigate the fraction of working water pumps by administrative region.
"""

# ╔═╡ 219d0f63-0f3d-43f7-86b2-632b5860c4ca
y = DataFrame(CSV.File(yfile))

# ╔═╡ e1bc52c5-d1fd-4f95-9556-a47fa638086a
md"""
Since I got rid of some of the samples that didn't make sense, the ones that I had (longitude, latitude) = (0, 0), I need to account for this in the $\mathbf{y}$ vector by using the `id` column
"""

# ╔═╡ b4efa4eb-ef53-43f8-8924-31cdf12f4472
id_set = intersect(y.id, df1.id)

# ╔═╡ f26c71e3-e00f-4bb1-9a8a-2542be992331
y1 = y[[y.id[i] in id_set for i = 1:length(y.id)], :]

# ╔═╡ be42d511-5f8c-4b9f-b630-82b6d719842d
md"""
There are actually three labels in this data: "Functional", "Non-Functional", and "Functional, but Needs Repairs." I want convert this to a binary problem for now (maybe I'll explore generalizing this later on), and so I'm going to map "Needs Repairs" to "Functional" which in turn will be $1$
"""

# ╔═╡ 17bdcd39-6268-4b46-a29c-8c2c3519a97f
label_map = Dict("functional" => 1,
				 "functional needs repair" => 1,
				 "non functional" => 0)

# ╔═╡ 71a0e20f-6140-4660-93d5-3760d34436ae
y1.status_group = map(x -> label_map[x], y1.status_group)

# ╔═╡ c630a88d-8d46-4857-beca-c91f96a52901
first(y1, 5)

# ╔═╡ bf91a72f-ba2a-4adb-abcf-162368f3949a
df2 = innerjoin(df1, y1, on=:id)

# ╔═╡ 2db8def5-91b1-45bc-b610-cf590a7cfcfd
function plot_water_pump_prob(df, region)
	y = df[df.region .== region, "status_group"]
	n = length(y)
	p = sum(y) / n
	
	b = bar([0, 1], [1-p, p], legend=false)
	xticks!(b, [0, 1])
	ylims!(b, (0, 1))
	title!("$(uppercasefirst(region))")
	annotate!(b, 1, p+0.1, text(L"\hat p \approx %$(round(p; digits=3))", 12))
end

# ╔═╡ a496c844-773a-4fef-8d73-8c3a27f6a166
begin
	plt_arr = []
	for region in sort(unique(df2.region))
		push!(plt_arr, plot_water_pump_prob(df2, region))
	end
	
	plot(plt_arr..., size=(1000, 800))
end

# ╔═╡ 23120c0d-cdb7-4a08-a9f6-ef3925858f34
md"""
It's interesting how varied the functionality distribution is across the regions. However, it's also noteworthy that there are a number of water pumps that have no population around it. IMO, I don't see why we would care about those water pumps. If they're not near any sort of servicing population then it's irrelevant whether it works or not.

I'll compute population distribution below.
"""

# ╔═╡ d78c876a-0909-465e-9601-2458a4202ba1
histogram(log.(df2.population), legend=false, title="Population Distribution",
          xlabel="Log(Population)")

# ╔═╡ 47756a1e-e65f-49b2-8af8-35c0fdd1cca1
sum(df2.population .== 0)

# ╔═╡ 2c7b4262-2024-41be-8d5a-130335936ea3
md"""
It seems like approximately $\frac{1}{3}$ of the data is a sample that does not service any particular population. Let's just drop those rows.
"""

# ╔═╡ d75a6759-e94d-4d4d-b9a4-a9937758a5e9
df3 = df2[df2.population .> 0, :];

# ╔═╡ ca59a81f-2e2c-4306-aa87-8c1a362921c2
begin
	plt_arr1 = []
	for region in sort(unique(df3.region))
		push!(plt_arr1, plot_water_pump_prob(df3, region))
	end
	
	plot(plt_arr1..., size=(1000, 800))
end

# ╔═╡ 97319a31-c30d-4f28-ae4e-ff68ac0b2907
md"""
Unsurprisingly the distribution has changed a little bit. In general it seems like the working proportion has gone up which makes sense to me.

Interestingly we also lost four regions who apparently all have servicing populations of zero. That seems very surprising to me that all of the wells in Dodoma (the capital region of Tanzania) has a servicing population of zero. That doesn't make much sense, but I'm also not sure how I can fix that. I suppose I could try to infer the population by looking at the village/area the water pump services, but I'm not sure how accurate Tanzania's census data is. That is something I'll have to investigate.

It looks like Tanzania did its last census in 2012 (apparently the next one is in 2022). The database is stored by the [National Bureau of Statistics](http://www.dataforall.org/CensusInfoTanzania/libraries/aspx/Home.aspx). Unfortunately I don't think using the census data is going to work. To get that data I would have to web scrape their database, and it's not clear the benefit will be worth the effort. For now, we'll just ignore all the samples that don't have any population and put this as a "Future Work" (like the region re-mapping).

One thing that may be valuable to explore though is that the above plot reflects the MLE, $\hat{p}$ of the probability of a pump working in a particular region. Of course we should reflect our uncertainty in that estimate. Therefore I think it would be wise to have the following mathematical construction: Let $R$ be the set of regions and $F \in \{0, 1\}$ be the event that a particular waterpump is functioning. We can therefore construct an empirical prior as follows:

```math
\begin{equation*}
	F_i \vert R \stackrel{iid}{\sim} \text{Beta}(a_r, b_r)
\end{equation*}
```

where $a_r$ and $b_r$ define the number of functioning and non-functioning water pumps in region $r$. Let's try plotting it for each of the regions to see what that distribution looks like.
"""

# ╔═╡ bb0d3044-8c4e-4506-9be0-f8633e3ac4c4
function plot_beta_distn(df, region)
	y = df[df.region .== region, "status_group"]
	a = sum(y .== 1)
	b = sum(y .== 0)
	
	p = plot(Beta(a, b), legend=false)
	title!("$(uppercasefirst(region))")
	xlabel!(p, L"p")
end

# ╔═╡ 6b8ae7dc-8e96-4aa3-a2a8-8953219a2790
begin
	plt_arr2 = []
	for region in sort(unique(df3.region))
		push!(plt_arr2, plot_beta_distn(df3, region))
	end
	
	plot(plt_arr2..., size=(1000, 800), link=:y)
end

# ╔═╡ 44772725-9b2e-459f-9920-433cf1cf6303
md"""
Unsurprisingly a number of the distribution have very tight probability estimates given the sheer number of samples associated with the region. However for others like Mwanza, by using the beta distribution it allows us to express our uncertainty about the value which I think will be helpful when we eventually build a Bayesian model. However, the claim that samples are iid in a particular region is probbaly not true. I imagine there are spatial correlations. We'll explore that more later.
"""

# ╔═╡ bbbf5e13-ad5b-4602-8962-b4c96a06d1ba
md"""
# Conclusion
This notebook is getting a bit long and I'm going to have to do some web-scraping to get the GDP data for my next analysis which I think is a completely separate effort and so I want to write some final thoughts about basic data transformations I did during this process so I can be more systematic in building a data transformation pipeline in the future

## Data Transformations
1. Removed data points that had (longitude, latitude) = (0, 0)
2. Removed samples that had a population of zero around the water pump
3. Identified the intersection between the `id` from the $\mathbf{X}$ and $\mathbf{y}$ and did an inner-join

## Final Thoughts
* To appropriately scale this work we're just going to use the regions provided in the data
  * As noted this is quite strange given that the Tanzanian government currently considers there to be $31$ administrative regions, but the amount of work to re-map the value based on coordinates would be way more than the benefit
* It is quite likely that the population values, particularly the ones listed as zero, are wrong. However the TZA census data is not particularly amenable to aligning with the subvillages and for many of them it seemed like it was last collected in 2002. If I continue in this vein that would be something to correct, but it's fine to just drop the zero rows for now.
* Clearly and unsurprisingly there are differences in water-pump functionality across the regions. I fit an iid Beta distribution for each of these regions, but I suspect that there are spatial correlations that we'll need to consider. Additionally, I want to explore other explanation for why there are differences. They could be economic as I inted to explore in the next notebook, but there are other plausible explanations as well. 
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PolygonOps = "647866c9-e3ac-4575-94e7-e3d426903924"
Shapefile = "8e980c4a-a4fe-5da2-b3a7-4b4b0353a2f4"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
CSV = "~0.8.5"
DataFrames = "~1.2.2"
Distributions = "~0.25.18"
LaTeXStrings = "~1.2.1"
Plots = "~1.22.4"
PolygonOps = "~0.1.2"
Shapefile = "~0.7.1"
StatsPlots = "~0.14.28"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra"]
git-tree-sha1 = "2ff92b71ba1747c5fdd541f8fc87736d82f40ec9"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.4.0"

[[Arpack_jll]]
deps = ["Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "e214a9b9bd1b4e1b4f15b22c0994862b66af7ff7"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.0+3"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "a4d07a1c313392a77042855df46c5f534076fab9"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.0"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[CSV]]
deps = ["Dates", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode"]
git-tree-sha1 = "b83aa3f513be680454437a0eee21001607e5d983"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.8.5"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "a325370b9dd0e6bf5656a6f1a7ae80755f8ccc46"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.7.2"

[[Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "31d0151f5716b655421d9d75b7fa74cc4e744df2"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.39.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[DBFTables]]
deps = ["Printf", "Tables", "WeakRefStrings"]
git-tree-sha1 = "3887db9932c2f9f159d28bfbe34f25597048eb80"
uuid = "75c7ada1-017a-5fb6-b8c7-2125ff2d6c93"
version = "0.2.3"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d785f42445b63fc86caa08bb9a9351008be9b765"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.2.2"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "9f46deb4d4ee4494ffb5a40a27a2aced67bdd838"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.4"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "ff7890c74e2eaffbc0b3741811e3816e64b6343d"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.18"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "463cb335fa22c4ebacfd1faba5fde14edb80d96c"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.5"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "29890dfbc427afa59598b8cfcc10034719bd7744"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.6"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "0c603255764a1fa0b61752d2bec14cfbd18f7fe8"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+1"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "c2178cfbc0a5a552e16d097fae508f2024de61a3"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.59.0"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "ef49a187604f865f4708c90e3f431890724e9012"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.59.0+0"

[[GeoInterface]]
deps = ["RecipesBase"]
git-tree-sha1 = "38a649e6a52d1bea9844b382343630ac754c931c"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "0.5.5"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "14eece7a3308b4d8be910e265c724a6ba51a9798"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.16"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "8a954fed8ac097d5be04921d595f741115c1b2ad"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+0"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "61aa005707ea2cebf47c8d780da8dc9bc4e0c512"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.4"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a4b12a1bd2ebade87891ab7e36fdbce582301a92"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.6"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "761a393aeccd6aa92ec3515e428c26bf99575b3b"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+0"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "34dc30f868e368f8a17b728a1238f3fcda43931a"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.3"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "5455aef09b40e5020e1520f551fa3135040d4ed0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+2"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "5a5bc6bf062f0f95e62d0fe0a2d99699fed82dd9"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.8"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "8d958ff1854b166003238fe191ec34b9d592860a"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.8.0"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "16baacfdc8758bc374882566c9187e785e85c2f0"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.9"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "c0e9e582987d36d5a61e650e6e543b9e44d9914b"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.7"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4dd403333bcf0909341cfe57ec115152f937d7d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.1"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "bfd7d8c7fd87f04543810d9cbd3995972236ba1b"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "1.1.2"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "b084324b4af5a438cd63619fd006614b3b20b87b"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.15"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "6841db754bd01a91d281370d9a0f8787e220ae08"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.22.4"

[[PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a193d6ad9c45ada72c14b731a318bedd3c2f00cf"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.3.0"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "69fd065725ee69950f3f58eceb6d144ce32d627d"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "01d341f502250e81f6fec0afe662aa861392a3aa"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.2"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "54f37736d8934a12a200edea2f9206b03bdf3159"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.7"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Shapefile]]
deps = ["DBFTables", "GeoInterface", "RecipesBase", "Tables"]
git-tree-sha1 = "1f4070fed3e779b4f710583f8dacd87397cd13b1"
uuid = "8e980c4a-a4fe-5da2-b3a7-4b4b0353a2f4"
version = "0.7.1"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "793793f1df98e3d7d554b65a107e9c9a6399a6ed"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.7.0"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8cbbc098554648c84f79a463c9ff0fd277144b6c"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.10"

[[StatsFuns]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "95072ef1a22b057b1e80f73c2a89ad238ae4cfff"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.12"

[[StatsPlots]]
deps = ["Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "eb007bb78d8a46ab98cd14188e3cec139a4476cf"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.28"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "019acfd5a4a6c5f0f38de69f2ff7ed527f1881da"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.1.0"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "fed34d0e71b91734bf0a7e10eb1bb05296ddbcd0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[WeakRefStrings]]
deps = ["DataAPI", "Random", "Test"]
git-tree-sha1 = "28807f85197eaad3cbd2330386fac1dcb9e7e11d"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "0.6.2"

[[Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "80661f59d28714632132c73779f8becc19a113f2"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.4"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "59e2ad8fd1591ea019a5259bd012d7aee15f995c"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.3"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─af7df31b-dafe-4ac7-9a26-958a7c2b3ebe
# ╠═a9710e38-25ef-11ec-2bea-c152dd5b1b5c
# ╠═a43cb588-ddd9-4cf2-a9cd-1a3c1e9d4005
# ╠═0b8f5add-00d2-49ec-b598-6ff34670d195
# ╠═040f381a-0cda-40d4-bec6-4f4a813adf5d
# ╠═0d59f5e4-de99-4aaf-9cd5-d47dfa3fd9b4
# ╟─256e435f-3471-4eb1-9d59-3db531168b42
# ╠═00b066f4-1224-4b6f-a335-a3d1da1123f4
# ╟─1e73a7c1-88cd-4766-b314-cdd5ca09a9f2
# ╟─169be54b-4562-4b1c-8cae-a27c07b054d1
# ╠═757b5ce2-63b8-4266-a00a-3b545b01c8ed
# ╠═ebfda391-25a7-41f1-b57f-964058c6d29c
# ╠═05dc6598-a645-45d0-a44f-d06229847d55
# ╟─ad9337fa-e7da-43eb-ad02-ca16bd520f47
# ╠═95a34546-90b2-4df3-8c26-cb2e1d8d3022
# ╟─190f5259-61aa-4c04-8661-98ec9ef919ef
# ╠═459e7e20-c65b-4d92-a45c-21d4e611df6f
# ╠═fb929142-d177-4685-b659-cb90b17cd605
# ╟─3bd619a7-7d54-47e0-a8df-ca117ba9ad35
# ╠═bd1e2ecf-dd98-4814-b161-190141b497d1
# ╠═cb4f4a27-8603-4994-9547-9cfa43f81123
# ╟─7e40284e-18d4-4009-8569-36ae24cec7cc
# ╟─5aac270a-3935-4338-9b1b-2fb1fdcef477
# ╠═db7e2740-2598-4eb7-a226-8188989b4562
# ╠═c576974a-1e1f-4dc0-bfe2-5ed25075b607
# ╠═63cef946-7932-433a-9b99-635cf80f5ab0
# ╟─77e9f470-1df9-481e-b1fa-063508fbd7b9
# ╟─9669a2cf-1417-4bc4-808d-858de9371d07
# ╠═906f44f9-403d-411d-943e-d873795887fe
# ╠═dab8667a-68b4-4bd4-96f6-dab0429cef73
# ╠═0e4f8f63-a2d5-438c-a580-e4a023467f4b
# ╠═a5c07da4-13a6-4a71-a187-aea28ea74add
# ╠═85305f5f-177f-4b2c-9c84-cf1708eb6779
# ╠═7587cc2a-64c7-40ab-af26-237eab25f41c
# ╟─212d38a9-1d7c-4661-9c3f-defb707a86b6
# ╠═0ffc9737-879a-48eb-8345-edaba4fb83f3
# ╠═4305c38e-4b50-49b8-97b7-66e7265279d6
# ╟─5f2428c8-3a0b-4001-910a-94ac3465fc16
# ╠═26a65919-7896-40b8-8fb1-813f409f406f
# ╠═d5f34bc0-b1ee-402b-8033-04bfdefa1980
# ╠═b088c955-8587-4128-b045-9c72c92acc9c
# ╠═2d3ed9dd-c97a-43cc-b572-64f7979bc0c1
# ╠═65c7a621-045d-4996-ad67-36dea8765167
# ╠═67bc4d3d-e660-4a7b-bd70-616e38774be8
# ╟─30c130da-0c1d-476a-9c18-f8b509c4db9e
# ╠═d3183657-d469-4257-a14a-b3584a16c18f
# ╠═7a733e15-8531-4213-aea8-220664a3bfb9
# ╠═79956456-d383-48ba-8083-6fe8195af521
# ╠═d9c7ddbf-5e78-4430-9d2d-cf283536fd47
# ╠═d8afca88-c70e-469f-b166-7e59eb635ae1
# ╟─0654a284-ac3a-4c96-8814-5d4496cca986
# ╟─e8d154a1-0df4-438e-b724-af1773c89f0f
# ╟─d58f6eed-8002-4363-820f-b959ff19fa8e
# ╠═219d0f63-0f3d-43f7-86b2-632b5860c4ca
# ╟─e1bc52c5-d1fd-4f95-9556-a47fa638086a
# ╠═b4efa4eb-ef53-43f8-8924-31cdf12f4472
# ╠═f26c71e3-e00f-4bb1-9a8a-2542be992331
# ╟─be42d511-5f8c-4b9f-b630-82b6d719842d
# ╠═17bdcd39-6268-4b46-a29c-8c2c3519a97f
# ╠═71a0e20f-6140-4660-93d5-3760d34436ae
# ╠═c630a88d-8d46-4857-beca-c91f96a52901
# ╠═bf91a72f-ba2a-4adb-abcf-162368f3949a
# ╠═2db8def5-91b1-45bc-b610-cf590a7cfcfd
# ╠═a496c844-773a-4fef-8d73-8c3a27f6a166
# ╟─23120c0d-cdb7-4a08-a9f6-ef3925858f34
# ╠═d78c876a-0909-465e-9601-2458a4202ba1
# ╠═47756a1e-e65f-49b2-8af8-35c0fdd1cca1
# ╟─2c7b4262-2024-41be-8d5a-130335936ea3
# ╠═d75a6759-e94d-4d4d-b9a4-a9937758a5e9
# ╠═ca59a81f-2e2c-4306-aa87-8c1a362921c2
# ╟─97319a31-c30d-4f28-ae4e-ff68ac0b2907
# ╠═bb0d3044-8c4e-4506-9be0-f8633e3ac4c4
# ╠═6b8ae7dc-8e96-4aa3-a2a8-8953219a2790
# ╟─44772725-9b2e-459f-9920-433cf1cf6303
# ╟─bbbf5e13-ad5b-4602-8962-b4c96a06d1ba
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
