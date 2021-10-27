### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ a9b329a2-2839-11ec-2fc4-393651b2e5e7
begin
	using CSV
	using DataFrames
	using DataFramesMeta
	using HTTP
	using Gumbo
	using Cascadia
	using Plots
	using Statistics
	using Shapefile
	using Dates
	using GLM
	using PlutoUI
end

# ╔═╡ b1211753-eb53-4912-ae0e-e22c865c4f45
TableOfContents(aside=true)

# ╔═╡ b3969d26-9443-436e-b0bf-3d2a3ecf087e
md"""
# Notebook Purpose
The purpose of this notebook is explore regional differences and develop plausible hypotheses that will ultimately be useful when developing our Bayesian model. In particular I want to understand:

1. Why are there water-pump functionality differences between regions?
2. Are there intra-regional differences (i.e., parts of a region have more functionality than others)?

To accomplish this goal, I will first implement the data transformations detailed in the first exploratory notebook, `geography.jl` and then I will start with my first hypothesis -- regional GDP partially explains inter-regional differences. To test this hypothesis I need to scrape these [GDP Tables](https://en.wikipedia.org/wiki/List_of_regions_of_Tanzania_by_GDP).
"""

# ╔═╡ 3f1cbe06-691a-438f-8c69-2fab55bf31cc
Xfile = "/home/zblanks/Documents/uva/fall21/bayesian-ml/project/data/X.csv";

# ╔═╡ 033acc63-7579-425b-990c-db21b1b63a3e
yfile = "/home/zblanks/Documents/uva/fall21/bayesian-ml/project/data/y.csv";

# ╔═╡ 9a87950c-e77d-4234-b692-c7cb3d5f9b18
function prep_labels(y)
	label_map = Dict("functional" => 1, "functional needs repair" => 1,
				     "non functional" => 0)
	
	@transform!(y, @byrow :status_group = label_map[:status_group])
	return y
end

# ╔═╡ 1d9f8d71-edb7-43be-964a-570252b3e8a6
function prep_data(Xfile, yfile)
	df = DataFrame(CSV.File(Xfile))
	y = DataFrame(CSV.File(yfile))
	
	df = @chain df begin
		@subset(:longitude .!= 0)
		@subset(:population .> 0)
	end
	
	y = prep_labels(y)
	df = innerjoin(df, y, on=:id)
	return df
end	

# ╔═╡ 1bada920-7f50-4dd2-bc07-05836bb37bd8
df = prep_data(Xfile, yfile);

# ╔═╡ 6ea2193d-9821-41cc-a2d8-5def1cdc9c56
md"""
# GDP Hypothesis
Now that the data is ready to go, I'm going to scrape TZA GDP data to test the hypothesis that part of inter-regional differences can be explained by GDP levels
"""

# ╔═╡ d76acb90-dd3d-4553-85ba-470f07c0b214
url = "https://en.wikipedia.org/wiki/List_of_regions_of_Tanzania_by_GDP"

# ╔═╡ e48d3c38-a72f-4e56-a765-6e7f1ccfe7df
r = HTTP.get(url);

# ╔═╡ 45ee554d-dbd6-4c72-b4dc-2667d250ff27
h = parsehtml(String(r.body));

# ╔═╡ 288a9bdf-0806-465c-b826-f0791eef5083
body = h.root[2];

# ╔═╡ 12d28b4e-21e6-4048-b478-7c39c8d04fee
s = eachmatch(Selector(".wikitable.sortable"), body)

# ╔═╡ eb06ae75-6ca5-404c-a2b6-29a6bdd85391
begin
	table = s[2][2]
	nrows = length(table.children)
end

# ╔═╡ 3bf3f190-4662-49c6-92e8-62a54eaa58b6
regions = [table[i][2][1][1].text for i = 2:(nrows - 1)]

# ╔═╡ 1f5aaef9-bc3e-4168-869a-0b8cdd20f382
regions1 = replace.(regions, "Region" => "")

# ╔═╡ 687dac4a-3dbf-4947-99b4-81705529543d
regions2 = rstrip.(regions1)

# ╔═╡ 3d152db3-586d-498f-826b-90a6dd99a450
gdp = [table[i][4][1].text for i = 2:(nrows - 1)]

# ╔═╡ cdc03eba-e092-441f-922e-0e58b6ee8e00
gdp1 = replace.(gdp, "," => "")

# ╔═╡ 1c31a154-ba53-43c5-a4f7-b9d8141f4ef9
gdp2 = parse.(Int, gdp1)

# ╔═╡ 87c59ff9-edde-403b-9af0-1cfd27e272b3
gdp_df = DataFrame(region=regions2, gdp=gdp2)

# ╔═╡ f3f2967b-481f-4f0f-9df6-63b92d07d08f
begin
	phat = @chain df begin
		groupby(:region)
		@combine(p = mean(:status_group))
		innerjoin(gdp_df, on=:region)
	end
end

# ╔═╡ 633d2c1a-c365-42e6-ac5d-102ea0522c56
scatter(phat.gdp, phat.p, legend=false, xlabel="GDP", 
	ylabel="Proportion of Working Pumps")

# ╔═╡ 2333115d-cfb3-4f31-81df-5979e97a2c5d
md"""
Interestingly, there does not appear to be much of a relationship between a particular region's GDP and the fraction of working pumps. I'll more precisely quantify it with a correlation coefficient
"""

# ╔═╡ deb14d1b-d0a2-4c6f-bcac-5ed974de305c
cor(phat.gdp, phat.p)

# ╔═╡ 77ee997c-d57a-4d8d-a40e-e0d92802b539
md"""
As we could physically intuit from the plot, there is basically not statistical relationship between a region's GDP and the corresponding fraction of working pumps. I think that's a surprising result, but it does mean that I have to look elsewhere. The next hypothesis class I want to explore relates to understanding spatial spatial relationships between working and broken pumps.
"""

# ╔═╡ 5f7647a2-9a6f-4f7d-9d8b-0038b3f80187
md"""
# Investigating Construction Year

Another feature that I think could be very powerful is looking at the year a particular well was built. However through cursory data views, it seems a number of samples are missing and so we'll need to determine what fraction are missing and then likely develop an imputation strategy.
"""

# ╔═╡ 05d124b8-25d7-400c-b44e-8e38e65afa31
histogram(df.construction_year, legend=false)

# ╔═╡ 72f5d5a7-1c94-42b6-af96-8eb53b6f3f5d
md"""
It looks like $(sum(df.construction_year .== 0)) samples have a missing year value. That's not as bad as I thought. I'm curious to see what the distribution looks like if subset the data ignoring the missing samples.
"""

# ╔═╡ 80bb72fa-4fec-475b-a562-67a67ab4517f
begin
	@chain df begin
		@subset(:construction_year .!= 0)
		@select(:construction_year)
		histogram(_.construction_year, legend=false)
	end
end

# ╔═╡ d0779051-9ba1-411a-ae2b-3820dc63770c
md"""
It's a skewed distribution (which isn't surprising), but what's more relevant is when the sample was recorded versus when the well was built. I'll need to construct a new feature that accounts for this and then visualize the distribution
"""

# ╔═╡ f3592098-a375-4c0f-9bb8-3f2ba42d4145
begin
	@chain df begin
		@subset(:construction_year .!= 0)
		@transform(:construction_year = Date.(:construction_year))
		@transform(:date_diff = :date_recorded .- :construction_year)
		@transform(:date_diff = Dates.value.(:date_diff))
		@subset(:date_diff .>= 0)
		@transform(:date_diff = log.(:date_diff))
		glm(@formula(status_group ~ date_diff), _, Binomial(), ProbitLink())
	end
end

# ╔═╡ c91c2453-3015-4173-b5a0-d388fa529c08
md"""
Doing a simple logistic regression, there appears to be a statistically significant relationship between the effective construction date for a particular well and the corresponding probability that it works or not. Specifically, I used the log-transformation of the effective construction date to assess this relationship because a marginal increase in one day is not a large effect size and I wanted to make the coefficient results clearer. According to the model, for every log-unit increase in the construction date, the odds of the well being functional decreases by approximately $(round(1-exp(-0.3), digits=2)*100)%. I think this is a pretty intuitive result; older things tend to have a lower probability of working. I think this will be a particularly useful feature in our Bayesian model and we'll have to consider how we want to express this relationship when constructing it.
"""

# ╔═╡ e2109442-3015-4ea8-8e02-d570c41f8a42
md"""
# Water-Pump Type and Source Relationship

Another variable of potential interest is whether there is any relationship with the type of water pump (e.g., "hand-pump", "communal standpipe", etc.) and the corresponding probability of it working. I could imagine it going either way, so I want to briefly explore this feature to see if it's worth including in the model and to gain more intuition about the problem space.

Additionally there are difference sources such as rivers, shallow wells and others. I will also investigate this feature.
"""

# ╔═╡ d97c6cd3-c108-4e29-99b7-e43260dea35c
begin
	@chain df begin
		groupby(:waterpoint_type)
		@combine(count=length(:waterpoint_type), p = mean(:status_group))
	end
end

# ╔═╡ 77f1c9f6-58bd-4f14-bff2-b7423b2af32b
md"""
Unsurprisingly there are differences between the group probabilities. However, clearly we should have more confidence in certain estimates than other. For example, there are only six instances of "dam" in the data and it has a $\hat p \approx 0.83$ which means five of the size samples worked. Our model should reflect our uncertainty about this class of waterpoint types. Conversely, if the sample is a "communal standpipe" there are over $20$k samples in the data, the $\hat p \approx 0.726$ is probably a good prior probability estimate of that class of water pump working.
"""

# ╔═╡ 65d8ef82-d138-4ea2-9b5c-04e1d8602e81
begin
	@chain df begin
		groupby(:source)
		@combine(count=length(:source), p = mean(:status_group))
	end
end

# ╔═╡ 519f1ba1-f01b-491b-989f-177203b412fc
md"""
Similar to the waterpoint type, it seems there are differences in probability of a particular well working depending on its source. For example, when the water is being sourced from a lake there is only a $16\%$ chance the particular well works. I suspect this is effectively a proxy for how many resources have been invested in the well.
"""

# ╔═╡ c72909fa-fb05-4624-8890-f2c1b6f22628
md"""
# Spatial Relationships with Water Pump Functionality
Now I want to explore if there is a spatial connection between pumps working in a particular region. My current hypothesis is that for a particular region, if a pump is not working then this decreases the probability that "nearby" pumps are also working. Similarly, if a specific pump is working then "nearby" pumps have a higher chance of also being functional.

To start, I'll just plot the pumps for a particular region and color code whether they're working or not. I think this can perhaps provide some visual intuition on whether this hypothesis is correct.
"""

# ╔═╡ cb5520a9-8c50-49e5-a8b0-d43b9b82ce17
shp_file = "/home/zblanks/Documents/uva/fall21/bayesian-ml/project/data/tza/tza.shp";

# ╔═╡ c1b73a83-83e1-45e1-a05d-f843a24a5303
shp = Shapefile.shapes(Shapefile.Table(shp_file));

# ╔═╡ 3a56ef47-6bd7-4cd1-8f13-96ccbb6e963c
md"""
To start with I'm going to focus on Dar es Salaam. Approximately 57% of its water pumps are working so it'll be interesting to see where those functioning and non-functioning pumps are located. Additionally, it may be relevant to point out that DES is on the south-eastern portion of TZA and its east coast actually faces out to the Indian Ocean.
"""

# ╔═╡ fe496eba-7605-40dc-98d0-5f0784b0d41c
begin
	p = plot(shp[2], color=nothing)
	
	des = @subset(df, :region .== "Dar es Salaam")	
	scatter!(p, des.longitude, des.latitude,
	  		 color=ifelse.(iszero.(des.status_group), "#E66100", "#5D3A9B"))
end

# ╔═╡ e9865b8e-c373-46c7-9174-2fc97f9f04e4
md"""
I can't get the legend to work right now, but Orange => 0 and Purple => 1. There may be something to my hypothesis, but to be able to test I'm going to need to do the following:

1. Compute a distance matrix $\mathbf{D}$ that measures the $L_2$ distance between all of the points in DES. 

2. Do a bit of literature review on the topic of spatial correlation. I'm absolutely positive I'm not the first person to encounter this problem.
"""

# ╔═╡ 2978d023-4d4a-4e7b-9c4e-e7b8f3b414b2
md"""
I think this analysis is interesting, but I won't include it in the final model for the following reasons:

1. I don't know how to construct iid samples when using the spatial information. It necessarily requires one to have context regarding other samples and this doesn't fit into the standard ML approach I intend to employ

2. I think we can capture a number of the spatial relationships by using a hierarchical Bayesin model.
"""

# ╔═╡ 86e00e6e-9954-4a7e-9690-51a55f627fbe
md"""
# Relevant Features
For the purpose of scoping this project appropriately (given the amount of work I have to do for other courses, I want to ensure that I'm building a tractable model that answers the bill. In particular, I want to ensure I'm looking at roughly speaking the "right" features. Right now I'm considering the following variables:

* Construction Year -- will transform it though
* Waterpoint Type
* Source
* Permit (maybe -- have to look at that in a moment)
* Payment (maybe -- also need to explore; this seems like it could be relevant)
* Log(Population)

And then constructing a hierarchy using region and possibly district. Therefore, I still need to investigate if there is any relationship for permit and payment schemes and the corresponding probability of the pump working. Afterwards, I then want to start building some initial models.
"""

# ╔═╡ e2828211-68c2-443d-abee-2ad14920b01b
md"""
## Permit Status
"""

# ╔═╡ 1c769eef-7631-482e-bd68-29b9db976fbb
begin
	@chain df begin
		groupby(:permit)
		@combine(count=length(:permit), p = mean(:status_group))
	end
end

# ╔═╡ 161890fb-d555-4c6d-bcbb-eded44a577c8
md"""
Interestingly there doesn't appear to be a very strong relationship with the permit status and the corresponding probability of it working. Moreover, it appears $1947$ samples are missing. It's possible I could use an imputation schema to infer their value, but I don't think it will be worth it given that this relationship doesn't appear to be that significant.
"""

# ╔═╡ fa77df19-9b38-48ca-916f-c6d05f8ba219
md"""
## Payment Status
"""

# ╔═╡ 8a8aae0d-326a-403f-a848-7d9523b69f8b
begin
	@chain df begin
		groupby(:payment_type)
		@combine(count=length(:payment_type), p = mean(:status_group))
	end
end

# ╔═╡ 2107b337-e737-43c9-aa4b-443ff02b129e
md"""
While it appears there are roughly seven payment schemes in the data, I think the results are relatively understandable -- when there is some sort of payment scheme (whether by bucket, monthly, annually, etc.) the corresponding probability of that source being functional is much higher than when there is no payment scheme or it is unknown. I think a simple transformation we can do with this variable is to condense it into a binary variable -- is there a payment scheme or not. This will help with limiting the creation of additional categorical variables while still capturing the essence of the insight.
"""

# ╔═╡ 4f381b45-1fd3-459e-a3c2-96849fc801b9
md"""
## Other Features
Finally I want to look at a few more features before wrapping up this particular notebook.
"""

# ╔═╡ a400d8e7-0878-42d3-85f1-396e038cdcd3
begin
	@chain df begin
		groupby(:quantity)
		@combine(count=length(:quantity), p = mean(:status_group))
	end
end

# ╔═╡ afa778be-550f-4351-b7a1-b4b3c6cc78c6
md"""
Unsurpringly, if a well is dry, only about $3\%$ work because why provide resources to maintain the source if there is not much water. I think this feature will be similar to payment scheme where we can collapse some of the categories while still capturing the key insight. In particular I think if I group "enough" and "seasonal" together, combine "dry" and "unknown" and keep "insufficient" by itself this will reduce the space down to three categories thus requiring only three indicators in the logistic regression model.
"""

# ╔═╡ 08f0724c-3c8a-4afb-bebb-821823ffaac5
begin
	@chain df begin
		groupby(:water_quality)
		@combine(count=length(:water_quality), p = mean(:status_group))
	end
end

# ╔═╡ bd402fdb-a763-4765-8b2a-4f9e13404204
md"""
It seems that the water quality feature won't provide much insight. Almost all the samples are "soft" and it seems to capture that prior probability that roughly $65\%$ of sources seem to be working. The remaining instances are likely just noise so I don't think there's much value in employing this feature in the model.
"""

# ╔═╡ 83a45448-acdc-43bd-98b7-33545214179a
begin
	@chain df begin
		groupby(:management_group)
		@combine(count=length(:management_group), p = mean(:status_group))
	end
end

# ╔═╡ fd304244-e9c1-49df-8a73-98f814e43e8c
md"""
Same as above, it seems that almost all of the samples are "user-group" and seem to roughly correspond with the prior so I don't think there will be much utility by including this feature.
"""

# ╔═╡ fd3e370b-09d2-454d-9e33-610e2d2de590
begin
	@chain df begin
		groupby(:extraction_type_group)
		@combine(count=length(:extraction_type_group), p = mean(:status_group))
	end
end

# ╔═╡ f2f9ee38-da67-4d0c-a86e-501c825f7111
md"""
In general, it appears there is little relationship between how the water is extracted and its corresponding probability of working. The exception to this observation seems to be "other" and groups that have smaller sample sizes (and thus are more prone to noise). I'm not particularly convinced this will be a valuable feature because I'm not sure how I would combine the various features to solidify this insight. Thus for the time being, I will ignore it.
"""

# ╔═╡ f5f870c3-64ca-4042-a443-a14683069883
begin
	@chain df begin
		groupby(:funder)
		@combine(count=length(:funder), p = mean(:status_group))
	end
end

# ╔═╡ fa62e0c8-0991-4d87-97be-cb48ac4c2076
md"""
There are over $1400$ categories for the funders. This is too high dimensional to work with and I don't think it'll provide much insight. I'm also going to ignore this feature.
"""

# ╔═╡ 2ca6cd16-e278-40f0-acce-600b41ee0a4a
md"""
Let summarize what we've learned from this section:

1. The water status of a well likely matters and we can condense this down into three variables (thereby limiting the number of new categorical features we have to create)

2. The extraction method, funder, and ownership group seem to have little relationship with the corresponding probability of the well being functional and so I won't include them in the model.
"""

# ╔═╡ 8ce4b880-666e-4b49-a6ac-3165dc82313a
md"""
# Conclusions

1. A well's construction date (unsurpringly) matters. We can use a transformation of this feature to ensure that we're capturing that insight in our model.

2. The source and type of water pump seems to have a relationship with the status of the pump

3. The existence of some sort of payment schema appears predictive of a water pump's status

4. The quantity of water at a particular location also appears relevant and we can condense this feature to limit the group of categorical varibales in the model.

5. There appear to be a number of irrelevant features such as the funder, extraction type, and others.
"""

# ╔═╡ a7e12049-8953-49e1-aff7-20c48e8396fe
md"""
## Data Pipeline

1. Removed data points that had (longitude, latitude) = (0, 0)
2. Removed samples that had a population of zero around the water pump
3. Inner-join "id" on $\mathbf{X}$ and $\mathbf{y}$
4. Drop samples where the construction year is zero
5. Compute the Log(Date Recorded - Construction Year)
6. Re-map the values for the payment schema to indicate the presence or non-presence of a plan
7. Re-map the values of the water quantity to be "enough", "insufficient", or "dry"
8. Drop irrelevant features
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
Cascadia = "54eefc05-d75b-58de-a785-1a3403f0919f"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DataFramesMeta = "1313f7d8-7da2-5740-9ea0-a2ca25f37964"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
Gumbo = "708ec375-b3d6-5a57-a7ce-8257bf98657a"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Shapefile = "8e980c4a-a4fe-5da2-b3a7-4b4b0353a2f4"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
CSV = "~0.8.5"
Cascadia = "~1.0.1"
DataFrames = "~1.2.2"
DataFramesMeta = "~0.9.1"
GLM = "~1.5.1"
Gumbo = "~0.8.0"
HTTP = "~0.9.16"
Plots = "~1.22.4"
PlutoUI = "~0.7.16"
Shapefile = "~0.7.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

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

[[Cascadia]]
deps = ["AbstractTrees", "Gumbo"]
git-tree-sha1 = "95629728197821d21a41778d0e0a49bc2d58ab9b"
uuid = "54eefc05-d75b-58de-a785-1a3403f0919f"
version = "1.0.1"

[[Chain]]
git-tree-sha1 = "cac464e71767e8a04ceee82a889ca56502795705"
uuid = "8be319e6-bccf-4806-a6f7-6fae938471bc"
version = "0.4.8"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "a325370b9dd0e6bf5656a6f1a7ae80755f8ccc46"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.7.2"

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

[[DataFramesMeta]]
deps = ["Chain", "DataFrames", "MacroTools", "Reexport"]
git-tree-sha1 = "29e71b438935977f8905c0cb3a8a84475fc70101"
uuid = "1313f7d8-7da2-5740-9ea0-a2ca25f37964"
version = "0.9.1"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

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
git-tree-sha1 = "dba1e8614e98949abfa60480b13653813d8f0157"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+0"

[[GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "f564ce4af5e79bb88ff1f4488e64363487674278"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.5.1"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "c2178cfbc0a5a552e16d097fae508f2024de61a3"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.59.0"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4c8c0719591e108a83fb933ac39e32731c7850ff"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.60.0+0"

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

[[Gumbo]]
deps = ["AbstractTrees", "Gumbo_jll", "Libdl"]
git-tree-sha1 = "e711d08d896018037d6ff0ad4ebe675ca67119d4"
uuid = "708ec375-b3d6-5a57-a7ce-8257bf98657a"
version = "0.8.0"

[[Gumbo_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "86111f5523d7c42da0edd85ef7999c663881ac1e"
uuid = "528830af-5a63-567c-a44a-034ed33b8444"
version = "0.10.1+1"

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

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "5efcf53d798efede8fee5b2c8b09284be359bf24"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.2"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

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

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

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

[[PlutoUI]]
deps = ["Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "4c8a7d080daca18545c56f1cac28710c362478f3"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.16"

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

[[ShiftedArrays]]
git-tree-sha1 = "22395afdcf37d6709a5a0766cc4a5ca52cb85ea0"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "1.0.0"

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

[[StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "1bc8cc83e458c8a5036ec7206a04d749b9729fe8"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.6.26"

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
# ╟─b1211753-eb53-4912-ae0e-e22c865c4f45
# ╟─b3969d26-9443-436e-b0bf-3d2a3ecf087e
# ╠═a9b329a2-2839-11ec-2fc4-393651b2e5e7
# ╠═3f1cbe06-691a-438f-8c69-2fab55bf31cc
# ╠═033acc63-7579-425b-990c-db21b1b63a3e
# ╠═9a87950c-e77d-4234-b692-c7cb3d5f9b18
# ╠═1d9f8d71-edb7-43be-964a-570252b3e8a6
# ╠═1bada920-7f50-4dd2-bc07-05836bb37bd8
# ╟─6ea2193d-9821-41cc-a2d8-5def1cdc9c56
# ╠═d76acb90-dd3d-4553-85ba-470f07c0b214
# ╠═e48d3c38-a72f-4e56-a765-6e7f1ccfe7df
# ╠═45ee554d-dbd6-4c72-b4dc-2667d250ff27
# ╠═288a9bdf-0806-465c-b826-f0791eef5083
# ╠═12d28b4e-21e6-4048-b478-7c39c8d04fee
# ╠═eb06ae75-6ca5-404c-a2b6-29a6bdd85391
# ╠═3bf3f190-4662-49c6-92e8-62a54eaa58b6
# ╠═1f5aaef9-bc3e-4168-869a-0b8cdd20f382
# ╠═687dac4a-3dbf-4947-99b4-81705529543d
# ╠═3d152db3-586d-498f-826b-90a6dd99a450
# ╠═cdc03eba-e092-441f-922e-0e58b6ee8e00
# ╠═1c31a154-ba53-43c5-a4f7-b9d8141f4ef9
# ╠═87c59ff9-edde-403b-9af0-1cfd27e272b3
# ╠═f3f2967b-481f-4f0f-9df6-63b92d07d08f
# ╠═633d2c1a-c365-42e6-ac5d-102ea0522c56
# ╟─2333115d-cfb3-4f31-81df-5979e97a2c5d
# ╠═deb14d1b-d0a2-4c6f-bcac-5ed974de305c
# ╟─77ee997c-d57a-4d8d-a40e-e0d92802b539
# ╟─5f7647a2-9a6f-4f7d-9d8b-0038b3f80187
# ╠═05d124b8-25d7-400c-b44e-8e38e65afa31
# ╟─72f5d5a7-1c94-42b6-af96-8eb53b6f3f5d
# ╠═80bb72fa-4fec-475b-a562-67a67ab4517f
# ╟─d0779051-9ba1-411a-ae2b-3820dc63770c
# ╠═f3592098-a375-4c0f-9bb8-3f2ba42d4145
# ╟─c91c2453-3015-4173-b5a0-d388fa529c08
# ╟─e2109442-3015-4ea8-8e02-d570c41f8a42
# ╠═d97c6cd3-c108-4e29-99b7-e43260dea35c
# ╟─77f1c9f6-58bd-4f14-bff2-b7423b2af32b
# ╠═65d8ef82-d138-4ea2-9b5c-04e1d8602e81
# ╟─519f1ba1-f01b-491b-989f-177203b412fc
# ╟─c72909fa-fb05-4624-8890-f2c1b6f22628
# ╠═cb5520a9-8c50-49e5-a8b0-d43b9b82ce17
# ╠═c1b73a83-83e1-45e1-a05d-f843a24a5303
# ╟─3a56ef47-6bd7-4cd1-8f13-96ccbb6e963c
# ╠═fe496eba-7605-40dc-98d0-5f0784b0d41c
# ╟─e9865b8e-c373-46c7-9174-2fc97f9f04e4
# ╟─2978d023-4d4a-4e7b-9c4e-e7b8f3b414b2
# ╟─86e00e6e-9954-4a7e-9690-51a55f627fbe
# ╟─e2828211-68c2-443d-abee-2ad14920b01b
# ╠═1c769eef-7631-482e-bd68-29b9db976fbb
# ╟─161890fb-d555-4c6d-bcbb-eded44a577c8
# ╟─fa77df19-9b38-48ca-916f-c6d05f8ba219
# ╠═8a8aae0d-326a-403f-a848-7d9523b69f8b
# ╟─2107b337-e737-43c9-aa4b-443ff02b129e
# ╟─4f381b45-1fd3-459e-a3c2-96849fc801b9
# ╠═a400d8e7-0878-42d3-85f1-396e038cdcd3
# ╟─afa778be-550f-4351-b7a1-b4b3c6cc78c6
# ╠═08f0724c-3c8a-4afb-bebb-821823ffaac5
# ╟─bd402fdb-a763-4765-8b2a-4f9e13404204
# ╠═83a45448-acdc-43bd-98b7-33545214179a
# ╟─fd304244-e9c1-49df-8a73-98f814e43e8c
# ╠═fd3e370b-09d2-454d-9e33-610e2d2de590
# ╟─f2f9ee38-da67-4d0c-a86e-501c825f7111
# ╠═f5f870c3-64ca-4042-a443-a14683069883
# ╟─fa62e0c8-0991-4d87-97be-cb48ac4c2076
# ╟─2ca6cd16-e278-40f0-acce-600b41ee0a4a
# ╟─8ce4b880-666e-4b49-a6ac-3165dc82313a
# ╟─a7e12049-8953-49e1-aff7-20c48e8396fe
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
