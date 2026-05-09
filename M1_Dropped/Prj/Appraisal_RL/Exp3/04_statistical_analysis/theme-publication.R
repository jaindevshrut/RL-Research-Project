## From https://rpubs.com/Koundy/71792

library(ggplot2)
library(gridExtra)

theme_Publication <- function(base_size=14, base_family="Helvetica") {
      library(grid)
      library(ggthemes)
      (theme_foundation(base_size=base_size, base_family=base_family)
       + theme(plot.title = element_text(face = "bold",
                                         size = rel(1.2), hjust = 0.5),
               text = element_text(),
               panel.background = element_rect(colour = NA),
               plot.background = element_rect(colour = NA),
               panel.border = element_rect(colour = NA),
               axis.title = element_text(face = "bold",size = rel(1)),
               axis.title.y = element_text(angle=90,vjust =2),
               axis.title.x = element_text(vjust = -0.2),
               axis.text = element_text(), 
               axis.line = element_line(colour="black"),
               axis.ticks = element_line(),
               panel.grid.major = element_line(colour="#f0f0f0"),
               panel.grid.minor = element_blank(),
               legend.key = element_rect(colour = NA),
               legend.position = "bottom",
               legend.direction = "vertical",
               ## legend.key.size= unit(0.2, "cm"),
               # legend.margin = unit(0, "cm"),
               ## legend.title = element_text(face="italic"),
               plot.margin=unit(c(10,5,5,5),"mm"),
               strip.background=element_rect(colour="#f0f0f0",fill="#f0f0f0"),
               strip.text = element_text(face="bold")
          ))
}

scale_fill_Publication <- function(...){
      library(scales)
      discrete_scale("fill","Publication",manual_pal(values = c("#36adbd","#fdb462","#7fc97f","#ef3b2c","#662506","#a6cee3","#fb9a99","#984ea3","#ffff33")), ...)

}
# 386cb0
# 66b5c0
# scale_fill_Publication <- function(...){
#   library(scales)
#   discrete_scale("fill","Publication",manual_pal(values = c("#fc8d62","#66c2a5")), ...)
#   
# }

# scale_colour_Publication <- function(...){
#       library(scales)
#       discrete_scale("colour","Publication",manual_pal(values = c("#386cb0","#fdb462","#7fc97f","#ef3b2c","#662506","#a6cee3","#fb9a99","#984ea3","#ffff33")), ...)
# 
# }

scale_colour_Publication <- function(...){
  library(scales)
  discrete_scale("colour","Publication",manual_pal(values = c("#fc8d62","#66c2a5")), ...)
  
}

## Examples:

## Scatter <- ggplot(mtcars, aes(mpg,disp,color=factor(carb))) + geom_point(size=3) + labs(title="Scatter Plot")

# grid.arrange(Scatter,(Scatter +scale_colour_Publication()+ theme_Publication()),nrow=1)

## Bar <- ggplot(mtcars, aes(factor(carb),fill=factor(carb))) + geom_bar(alpha=0.7) + labs(title="Bar Plot")

## grid.arrange(Bar,(Bar + scale_fill_Publication() +theme_Publication()),nrow=1)
