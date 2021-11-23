################################################################################
# EAS451/585
# Lab 2: Multispectral Image Classification in R
# Using Supervised and Unsupervised Classifications
#
#
# Fall 2021
################################################################################

library(RStoolbox)
library(raster)
library(RStoolbox) 
library(raster)
library(tidyverse)
library(sf)
library(rpart)
library(rpart.plot)
library(rasterVis)
library(mapedit)
library(mapview)
library(caret)
library(forcats)
library(patchwork)
library(cluster)
library(randomForest)


# Read in all the band data, these are individual bands from a LandSat8 image

# Bands are as follows:

# Band 1	Coastal aersol
# Band 2	Blue
# Band 3	Green
# Band 4	Red
# Band 5	Near Infrared (NIR)
# Band 6	Shortwave Infrared (SWIR) 1
# Band 7	Shortwave Infrared (SWIR) 2
# Band 8	Panchromatic
# Band 9	Cirrus
# Band 10	Thermal Infrared (TIRS) 1
# Band 11	Thermal Infrared (TIRS) 2

band1 <- raster("data/band1.tif")
band2 <- raster("data/band2.tif")
band3 <- raster("data/band3.tif")
band4 <- raster("data/band4.tif")
band5 <- raster("data/band5.tif")
band6 <- raster("data/band6.tif")
band7 <- raster("data/band7.tif")
band8 <- raster("data/band8.tif")
band9 <- raster("data/band9.tif")
band10 <- raster("data/band10.tif")
band11 <- raster("data/band11.tif")

# Look at some of the bands, notice any variation or bands that stand out
plot(band1)
plot(band10)
plot(band5)

# Make notes of the band value ranges
# Check the resolutions
bands <- c(band1, band2, band3, band4, band5, band6, 
           band7, band8, band9, band10, band11)

for (i in bands) {
  check <- res(i)
  print(check)
}

# Be aware that in order to stack images, the resolutions must match
# to do this we will 'aggregate()' the higher resolution (lower number)
# to the lower resolution (larger number)
# fact = 2, in this case is a modifying factor meaning that we are multiplying the
# current resolution by 2. If we wanted to divide it by 2 we can use 1/2, or 0.5
band8 <- aggregate(band8, fact = 2)

for (i in bands) {
  check <- res(i)
  print(check)
}

# Make sure that all bands have the same resolution, aggregate as necessary
# once complete, lets stack the bands into one image
image <- stack(band1, band2, band3, band4, band5, band6, 
               band7, band8, band9, band10, band11)

# Check to make sure that the image now has all 11 bands as layers
# this will return the number of layers
nlayers(image) 

# How many layers should we have?

# Now lets check to make sure our spatial reference is still valid, otherwise 
# our image is lost spatially
# What coordinate system is used?

# Often the 'zone' is accompanied by either an S or N, what would these indicate?
crs(image)

# Confirm that the resolution of the image matches the aggregated resolution
res(image)

# Now that we have confirmed the necessary information, and created the image
# we can plot using band combinations, lets try a:
# True Color Composite 
# False Color Composite

# True Color Composite
par(col.axis = "white", col.lab = "white", tck = 0)
plotRGB(image, r = 4, g = 3, b = 2, axes = TRUE, 
        stretch = "lin", main = "True Color Composite")
box(col = "white")

# What bands are used in the True Color Composite? Refer to the band list at 
# the top ex: Band 9 (Cirrus)

# False Color Composite
par(col.axis = "white", col.lab = "white", tck = 0)
plotRGB(image, r = 5, g = 4, b = 3, axes = TRUE, stretch = "lin",
        main = "False Color Composite")
box(col = "white")

###############################################################################

#  Since we have create the image, we may wish to remove some of the extra files
# for the sake of memory space
for (i in bands) {
  rm(i)
}

# We can use a general 'garbage collection' to free up some space as well that 
# may be occupied
gc()

################################################################################

# What bands are used in the False Color Composite? Again, refer to the band list 
# (number, name) ex: Band 9 (Cirrus)

# What appears to be the main feature of the False Color Composite? (In RED)


# By observing our composites we can see a large amount of vegetation
# Given our available bands, we can derive an NDVI (Normalized Difference 
# Vegetation Index)
# Recall that NDVI scales from -1 to +1, with +1 indicating more vegetation 
# cover
# These values are largely driven by pigments in vegetation measured by the 
# bands used
ndvi <- (image[[5]] - image[[4]]) / (image[[5]] + image[[4]])

# What bands are used in the NDVI calculation? Why these bands? Recall the 
# Vegetation Spectrum

# lets observe the ndvi result

# minimum
min(ndvi@data@values, na.rm = T)

# maximum
max(ndvi@data@values, na.rm = T)

# standard deviation
sd(ndvi@data@values, na.rm = T)

# summary
summary(ndvi)

summary(ndvi@data@values)

# We can plot the NDVI as well
as(ndvi, "SpatialPixelsDataFrame") %>% 
  as.data.frame() %>%
  ggplot(data = .) +
  geom_tile(aes(x = x, y = y, fill = layer)) +
  theme(axis.text = element_blank(),
        axis.ticks = element_blank(),
        panel.background = element_blank(),
        panel.grid.minor = element_blank()) +
  labs(title = "NDVI for Calgary, Alberta", 
       x = " ", 
       y = " ") +
  scale_fill_gradient(high = "#CEE50E", 
                      low = "#087F28",
                      name = "NDVI")

# Feel free to use other color gradients and plotting schemes, be as creative 
# as you wish

# Do the higher NDVI values match up with any thing from the True Color or 
# False Color Composites?

# We are now going to attempt a Supervised Classification to try and classify 
# each pixel of the image into various classes

# As this is a Supervised Classification we will first need to create a 
# training dataset


# The below line will open up RGB image in the plotting window as an interactive
# image. You can pan around with the hand and zoom in with the scroll wheel. 
# Get familiar with the controls and movement before moving on. Once comfortable,
# select the option from the left menu for drawing a 'polygon', this will allow
# you to click and define a polygon You are clicking to place the vertices, 
# so be aware that it will be drawing straight lines between points. Once you
# have defined a polygon click on the first point to close it, then you can 
# click 'cancel' to clear the tool and move to the next area Please only do one
# class at a time, meaning that if you are defining agriculture, just draw all
# agriculture polygons. Once you have draw several polygons for one class, 
# select 'done' in the bottom right

# We need to run an individual instance of the below line for each class we 
# want. Please try to create the following classes:

# Urban
# Water
# Agriculture
# Other Vegetation

# Be aware that it may be beneficial to use the False Color Composite 
# for some classes such as water
# plotting the false color composite in mapview
# points <- viewRGB(image, r = 5, g = 4, b = 3) %>% 
#   editMap()


# agriculture -------------------------------------------------------------
points_agriculture <- viewRGB(image, r = 4, g = 3, b = 2) %>% 
  editMap()

saveRDS(points_agriculture, file = "data/points_agriculture.rds")

# Rename column with agriculture geometries
agriculture <- points_agriculture$finished$geometry %>% 
  st_sf() %>% 
  mutate(class = "agriculture", id = 1)


# Urban -------------------------------------------------------------------
points_urban <- viewRGB(image, r = 4, g = 3, b = 2) %>% 
  editMap()

saveRDS(points_urban, file = "data/points_urban.rds")

# Rename column with urban geometries
urban <- points_urban$finished$geometry %>% 
  st_sf() %>% 
  mutate(class = "urban", id = 2)

# Water -------------------------------------------------------------------
points_water <- viewRGB(image, r = 4, g = 3, b = 2) %>% 
  editMap()

saveRDS(points_water, file = "data/points_water.rds")

# Rename column with water geometries
water <- points_water$finished$geometry %>% 
  st_sf() %>%
  mutate(class = "water", id = 3)

# Vegetation -------------------------------------------------------------------
points_vegetation <- viewRGB(image, r = 4, g = 3, b = 2) %>% 
  editMap()

saveRDS(points_vegetation, file = "data/points_vegetation.rds")

# Rename column with vegetation geometries
veg <- points_vegetation$finished$geometry %>% 
  st_sf() %>%
  mutate(class = "vegetation", id = 4)

# Loading a point file ----------------------------------------------------

# If you missed one of your processes, load the point file with the next
# instruction:
# readRDS(file = "points_water.rds")

# Other vegetation will cover things such as shrublands, forests, or grasslands 
# that are possible not as vibrant green as agriculture

# We will now combine all the collected points into on training dataset
training_points <- rbind(agriculture, veg, water, urban)

# You can write these points as a shapefile with: 
write_sf(training_points, 
         "data/calgary_training_points.shp",
         driver = "ESRI shapefile")

# To be read back in later with: 
# training_points <- st_read("calgary_training_points.shp")

# Lets check our distribution of points for the training dataset
# Read in the city boundary for calgary
city_boundary <- st_read("data/CityBoundary.geojson", quiet = TRUE)

# create a map looking at just the distribution of polygons
polygons_distribution <- ggplot() +
  geom_sf(data = city_boundary, fill = "light gray", color = NA) +
  geom_sf(data = training_points, size = 0.5) +
  labs(title = "Distribution of\nclassification points") +
  theme(panel.background = element_blank(), axis.ticks = element_blank(), 
        axis.text = element_blank())

# create a map looking at the distribution of polygons by classification type
polygons_categories <- ggplot() +
  geom_sf(data = city_boundary, fill = "light gray", color = NA) +
  geom_sf(data = training_points, aes(fill = class), size = 0.5) +
  scale_fill_viridis_d() +
  labs(title = "Classification points by land use") +
  theme(panel.background = element_blank(), axis.ticks = element_blank(), 
        axis.text = element_blank())

# Plot side by side
polygons_distribution + 
  polygons_categories + 
  plot_layout(ncol = 2)

#  See how your distribution is allocated, if you feel as though it is biased to 
# one section or not well distributed you may wish to redo the training data
# specification

# Now we will extract the spectral data or band data for our training points
# first convert to a spatial point format
training_points <- as(training_points, 'Spatial')

#  Extract values to a data frame
df <- raster::extract(image, training_points) #%>%
  # round()

# We should now have a matrix of band values for each point
head(df)

# Try exploring the data through plotting
# remember you id numbers from the sections above
profiles <- df %>% 
  as.data.frame() %>% 
  cbind(., training_points$id) %>% 
  rename(id = "training_points$id") %>% 
  na.omit() %>% 
  group_by(id) %>% 
  summarise(band1 = mean(band1),
            band2 = mean(band2),
            band3 = mean(band3),
            band4 = mean(band4),
            band5 = mean(band5),
            band6 = mean(band6),
            band7 = mean(band7),
            band8 = mean(band8),
            band9 = mean(band9),
            band10 = mean(band10),
            band11 = mean(band11)) %>% 
  mutate(id = case_when(id == 1 ~ "agriculture",
                        id == 2 ~ "urban",
                        id == 3 ~ "water",
                        id == 4 ~ "other vegetation"
                        )) %>% 
  as.data.frame()

head(profiles)

# We can now plot the profiles to see if they are indeed unique and distinguishable

profiles %>% 
  select(-id) %>% 
  gather() %>% 
  mutate(class = rep(c("agriculture", "urban", "water", "other vegetation"),
                     11)) %>% 
  ggplot(data = ., aes(x = fct_relevel(as.factor(key),
                                       levels = c("band1", "band2", 
                                                  "band3", "band4",
                                                  "band5", "band6",
                                                  "band7", "band8",
                                                  "band9", "band10",
                                                  "band11")), y = value, 
                       group  = class, color = class)) +
  geom_point(size = 2.5) +
  geom_line(lwd = 1.2) +
  scale_color_manual(values=c('lawngreen', 'burlywood', 'lightblue', 'darkgreen')) +
  labs(title = "Spectral Profile from Landsat 8 Imagery",
       x = "Bands",
       y = "Reflectance") +
  #scale_y_continuous(limits=c(5000, 15000)) +
  theme(panel.background = element_blank(),
        panel.grid.major = element_line(color = "gray", size = 0.5),
        panel.grid.minor = element_line(color = "gray", size = 0.5),
        axis.ticks = element_blank())

#Another way to assess this is through a density plot, note any severe overlap between classes at each band
#The mean values will also indicate if there is a large degree of overlap between classes

profiles %>% 
  select(-id) %>% 
  gather() %>% 
  mutate(class = rep(c("agriculture", "urban", "water", "other vegetation"), 11)) %>% 
  ggplot(., aes(x=value, group=as.factor(class), fill=as.factor(class))) + 
  geom_density(alpha = 0.75) + 
  geom_vline(data = . %>% group_by(class) %>% summarise(grp.mean = mean(value)),
             aes(xintercept=grp.mean, color = class), linetype="dashed", size=1) +
  scale_fill_manual(values=c('lawngreen', 'burlywood', 'lightblue', 'darkgreen'),
                    name = "class") +
  scale_color_manual(values=c("black", "red", "orange", "yellow")) +
  theme(panel.background = element_blank(),
        panel.grid.major = element_line(color = "gray", size = 0.5),
        panel.grid.minor = element_line(color = "gray", size = 0.5),
        axis.ticks = element_blank()) +
  labs(x = "Reflectance Value",
       y = "Density",
       title = "Density histograms of spectral profiles",
       subtitle = "Vertical lines represent mean group reflectance values")

#Note the similarities in overlap between the density plot and the spectral profile. These overlapping
#classes may prove to be difficult to distinguish via the classification.

#Now we can move onto classifying thw image by training the model
#combine classes and extracted values into a dataframe
df <- data.frame(training_points$class, df)
#we then use the rpart() to train the model
model.class <- rpart(as.factor(training_points.class)~., data = df, method = 'class')

#We can plot a decision tree resulting from the training
rpart.plot(model.class, box.palette = 0, main = "Classification Tree")

#Now lets run the prediction from the model for the entire image
pr <- predict(image, model.class, type ='class', progress = 'text') %>% 
  ratify()
#set the levels to our selected classes
levels(pr) <- levels(pr)[[1]] %>%
  mutate(legend = c("agriculture","urban","water", "other vegetation"))

#And plot()
levelplot(pr, maxpixels = 1e6,
          col.regions = c('lawngreen', 'burlywood', 'lightblue', 'darkgreen'),
          scales=list(draw=FALSE),
          main = "Supervised Classification of Imagery")

#Do the results seem reasonable? Why or Why not? Where are areas or conflict?
#We can check the results using a confusion matrix to compare the predicted values to the ground-truth points

test <- raster::extract(pr, training_points) %>% 
  as.data.frame() %>% 
  rename(id = ".")

testProbs <- data.frame(
  obs = as.factor(training_points$id),
  pred = as.factor(test$id)
) %>% 
  mutate(correct = ifelse(obs == pred, 1, 0))

confMatrix <- confusionMatrix(testProbs$obs, testProbs$pred)
confMatrix

#Review the results, what class or classes tend to be misrepresented or misclassified? The rows and columns of the confusion
#matrix should provide this information, as well as the sensitivity analysis
#What is your overall accuracy?

#How to interpret a confusion matrix in R 
#https://www.journaldev.com/46732/confusion-matrix-in-r

#End of Supervised Image Classification

###############################################################################################################################
###############################################################################################################################
#This below section for the EAS451 class of Fall 2021 is optional
#Unsupervised Image Classification

#For this section we are going to be completely hands off in the classification process
#The idea is that the classification method will find statistically distinct pixels or clusters of pixels
#and use those to define numbered classes, this will be refined multiple times to give the fewest number of
#distinct classes as possible.

#We will use 3 different methods, Kmeans, Clara Clustering and Random Forest

#We need the original image data that we created above the image stack from line 62
#we will then create a matrix of values for the classification
## returns the values of the raster dataset and write them in a matrix. 
v <- getValues(image)
i <- which(!is.na(v))
v <- na.omit(v) #remove NA values because they cannot be classified

## kmeans classification 
#https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/kmeans
E <- kmeans(v, 4, iter.max = 100, nstart = 10)
kmeans_raster <- raster(image)
kmeans_raster[i] <- E$cluster
plot(kmeans_raster)

#In the kmeans() function we are specificying 10 intial centers or clusters, runninig 100 iterations and 10 intial configurations
#plot the results and review

## clara classification 
#https://www.rdocumentation.org/packages/cluster/versions/2.1.2/topics/clara
clus <- clara(v,4,samples=500,metric="manhattan",pamLike=T)
clara_raster <- raster(image)
clara_raster[i] <- clus$clustering
plot(clara_raster, col = topo.colors(5)) #try other colors as well

## unsupervised randomForest classification using kmeans
#https://www.rdocumentation.org/packages/randomForest/versions/4.6-14/topics/randomForest
vx<-v[sample(nrow(v), 500),]
rf = randomForest(vx)
rf_prox <- randomForest(vx,ntree = 1000, proximity = TRUE)$proximity

E_rf <- kmeans(rf_prox, 4, iter.max = 100, nstart = 10)
rf <- randomForest(vx,as.factor(E_rf$cluster),ntree = 500)
rf_raster<- predict(image,rf)
plot(rf_raster)


#The three classifications are stacked into one layerstack and plotted for comparison.

class_stack <- stack(kmeans_raster,clara_raster,rf_raster)
names(class_stack) <- c("kmeans","clara","randomForest")

par(mfrow = c(3,1))
plot(class_stack)

#Ideally we would then test the accuracy against the ground truth file, however that would require the manual merging
#of the unsupervised classes and then assessing accuracy wihch for the sake of this assignment and time boundaries is 
#not required
