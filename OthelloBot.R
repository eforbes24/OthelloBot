## Eden Forbes
## OthelloBot
## COGS 319 

#### 1) CONFIGURE HOPPER AND LOAD PACKAGES ####

Sys.setenv("CUDA_VISIBLE_DEVICES"="1")

library(keras)
library(dplyr)

#### 2) GAME ENGINE ####
## Borrowed & Edited from Jack Davis, www.stats-et-al.com"
## The functions below are written for an interactive game of Othello between
## two human players. However, the inputs and outputs of some of the functions 
## can be used to create the datasets necessary for training OthelloBot (see
## below)

set.board = function(){
  board <- matrix(".", nrow=8, ncol=8)
  board[4,4] = "W"
  board[5,5] = "W"
  board[4,5] = "B"
  board[5,4] = "B"
  return(board)
}

look.to = function(board, position, direction){
  stones = character(0)
  xlist = numeric(0)
  ylist = numeric(0)
  this_x = position[2]
  this_y = position[1]
  if(!(direction %in% c("N","NE","E","SE","S","SW","W","NW")))
  {
    print("!!!! ERROR !!!! Not an allowable direction")
    return("!!!! ERROR !!!! Not an allowable direction")
  }
  if(!(this_x > 0 & this_x <= ncol(board) & this_y > 0 & this_y <= nrow(board)))
  {
    print("!!!! ERROR !!!! Position is out of bounds")
    return("!!!! ERROR !!!! Position is out of bounds")
  }
  if(direction == "N"){  xstep = 0;	ystep = -1}
  if(direction == "NE"){ xstep = 1;	ystep = -1}
  if(direction == "E"){  xstep = 1;	ystep = 0}
  if(direction == "SE"){ xstep = 1;	ystep = 1}
  if(direction == "S"){  xstep = 0;	ystep = 1}
  if(direction == "SW"){ xstep = -1;	ystep = 1}
  if(direction == "W"){  xstep = -1;	ystep = 0}
  if(direction == "NW"){ xstep = -1;	ystep = -1}
  while(this_x > 0 & this_x <= ncol(board) & this_y > 0 & this_y <= nrow(board))
  {
    stones = c(stones, board[this_y,this_x])
    xlist = c(xlist, this_x)
    ylist = c(ylist, this_y)
    this_x = this_x + xstep
    this_y = this_y + ystep
  }
  ### Output the stones/spaces, x coords, or y coords of the observed locations
  #if(output_form == "stones"){output = stones}
  #if(output_form == "xlist"){output = xlist}
  #if(output_form == "ylist"){output = ylist}
  #output = output[-1]
  #if(output_form == "stones"){output = paste(output, collapse = "")}
  stones = stones[-1]
  stones = paste(stones, collapse= "")
  xlist = xlist[-1]
  ylist = ylist[-1]
  output = list(stones=stones, xlist=xlist, ylist=ylist)
  return(output)
}

look.around = function(board, position)
{
  results = rep("",8)
  count = 1
  for(this_direction in c("N","NE","E","SE","S","SW","W","NW"))
  {
    results[count] = look.to(board, position, this_direction)$stones
    count = count + 1
  }
  return(results)
}

legal.look = function(player, look)
{
  look = strsplit(look, "")[[1]]
  if(length(look) < 2){ return(0)}
  Nenemies = 0
  enemy_chain = TRUE
  while(Nenemies < length(look) & enemy_chain)
  {
    examined_piece = look[Nenemies + 1]
    if(examined_piece %in% c(player,"."," ","#"))
    {
      enemy_chain = FALSE  ## If it's not an enemy, stop looking
    }
    else
    {
      Nenemies = Nenemies + 1 ## If it is an enemy, iterate and keep looking
    }
  }
  if(Nenemies == length(look)){return(0)}
  examined_piece = look[Nenemies + 1]
  if(examined_piece == player)
  {
    return(Nenemies)
  }
  return(0)
}

legal.directions = function(player, around)
{
  legal_looks = rep(FALSE,8)
  for(k in 1:length(around))
  {
    this_look = around[k]
    legal_looks[k] = legal.look(player, look=this_look)
  }
  directions = c("N","NE","E","SE","S","SW","W","NW")
  return(directions[legal_looks > 0])
}

which.legal = function(board,player)
{
  Nx = ncol(board)
  Ny = nrow(board)
  legal_board = matrix(FALSE,nrow=Ny, ncol=Nx)
  for(y in 1:Ny)
  {
    for(x in 1:Nx)
    {
      if(board[y,x] == ".")
      {
        around = look.around(board, position=c(y,x))
        Ndirections = legal.directions(player,around)
        if(length(Ndirections) >= 1){legal_board[y,x] = TRUE}
      }
    }
  }
  return(legal_board)
}

play.move = function(board, this_player, position)
{
  if(board[position[1],position[2]] != ".")
  {
    print("Stone must be placed on an empty place on the board")
    return(board)
  }
  around = look.around(board, position)
  legal_directions = legal.directions(player = this_player, around)
  new_board = board
  for(this_direction in legal_directions)
  {
    this_look = look.to(board, position, this_direction)
    x_to_change = this_look$xlist #, output_form="xlist")
    y_to_change = this_look$ylist #, output_form="ylist")
    stones_to_change = this_look$stones
    Nenemies = legal.look(player = this_player, look=stones_to_change)
    for(k in 1:Nenemies)
    {
      new_board[y_to_change[k],x_to_change[k]] = this_player
    }
  }
  if(length(legal_directions > 0))
  {
    new_board[position[1],position[2]] = this_player
  }
  return(new_board)
}

#### 3) DATASET CREATION FUNCTIONS ####

## The functions below are those written for this project. Each has a short 
## description saying what it does in the greater script, and together they 
## serve as a way of reformatting the output of the game engine above to 
## useful datasets for the model.

## legal.moves gives the legal moves for the player given the state of board. 
## Additionally adds the move probabilities generated by the model for each 
## space and normalizes the probabilities of those that are legal. Returns a 
## dataframe of the legal moves, their normalized probabilities, and labels of 
## the corresponding space's location on the board. 

legal.moves = function(board, move, player){
  legal.board <- which.legal(board, player)
  legal.board <- as.vector(legal.board)
  df <- data.frame(c(1:64))
  names(df)[1] <- "position"
  df$legal <- legal.board
  move <- array_reshape(move, 64)
  df$move.prob <- move
  df <- df %>% 
    filter(legal == TRUE) %>% 
    mutate(normalized = (move.prob + 1)/sum(move.prob + 1))
  return(df)
}

## model.board translates the game engine board format, which is given in 
## strings, to the CNN board format which is in digits. It also changes the 
## array to the format used by the model. 

model.board = function(board, player){
  if(player == "W"){
    board[board == "W"] <- 1
    board[board == "B"] <- -1
    board[board == "."] <- 0
  } else {
    board[board == "W"] <- -1
    board[board == "B"] <- 1
    board[board == "."] <- 0
  }
  board <- apply(board, 2, as.numeric)
  board <- array_reshape(board, c(1,8,8,1))
  return(board)
}

## determine.winner is a seemingly long function that really just determines
## which sets of moves and initial boards are coded positively and which are
## coded negatively for the model during training depending on which 
## player won the game. 

determine.winner = function(board.digit, player, inputs.W, inputs.B, moves.W,
                            moves.B, outputs.W, outputs.B){
  if(player == "W"){
    if(sum(board.digit) > 0){
      winning.inputs <- inputs.W
      losing.inputs <- inputs.B
      winning.moves <- moves.W
      losing.moves <- moves.B
      winning.outputs <- outputs.W
      losing.outputs <- outputs.B
      results <- list(winning.inputs, losing.inputs, winning.moves,
                      losing.moves, winning.outputs, losing.outputs)
      } else {
        winning.inputs <- inputs.B
        losing.inputs <- inputs.W
        winning.moves <- moves.B
        losing.moves <- moves.W
        winning.outputs <- outputs.B
        losing.outputs <- outputs.W
        results <- list(winning.inputs, losing.inputs, winning.moves,
                        losing.moves, winning.outputs, losing.outputs)
      }
    } else {
      if(sum(board.digit) > 0){
        winning.inputs <- inputs.B
        losing.inputs <- inputs.W
        winning.moves <- moves.B
        losing.moves <- moves.W
        winning.outputs <- outputs.B
        losing.outputs <- outputs.W
        results <- list(winning.inputs, losing.inputs, winning.moves,
                        losing.moves, winning.outputs, losing.outputs)
        } else {
          winning.inputs <- inputs.W
          losing.inputs <- inputs.B
          winning.moves <- moves.W
          losing.moves <- moves.B
          winning.outputs <- outputs.W
          losing.outputs <- outputs.B
          results <- list(winning.inputs, losing.inputs, winning.moves,
                          losing.moves, winning.outputs, losing.outputs)
        }
    }
  return(results)
}

## play.game simulates a full game of Othello and returns the initial board
## states, moves, and resulting board states for both the winner and the loser
## of the game. 

play.game = function(){
  ## Set the board and the running variables, WHITE goes first
  inputs.W <- array(data = NA, dim = c(30, 8, 8, 1))
  inputs.B <- array(data = NA, dim = c(30, 8, 8, 1))
  outputs.W <- array(data = NA, dim = c(30, 8, 8, 1))
  outputs.B <- array(data = NA, dim = c(30, 8, 8, 1))
  moves.W <- array(data = NA, dim = c(30, 64))
  moves.B <- array(data = NA, dim = c(30, 64))
  board <- set.board()
  player <- "W"
  i.W <- 1
  i.B <- 1
  ## Play the game until whoever's turn it is can no longer play
  while(TRUE %in% which.legal(board, player)){
      ## Give the network the board and get back the output for PLAYER
      ## Save the board and the move to the running variables
      board.digit <- model.board(board, player)
      if(player == "W"){
        inputs.W[i.W,,,] <- board.digit
      } else {
        inputs.B[i.B,,,] <- board.digit
      }
      move <- model %>% predict(board.digit)
      ## Take legal moves and renormalize probabilities to sum to 1. 
      ## Pick randomly based on those probabilities
      good.moves <- legal.moves(board, move, player)
      if(length(c(good.moves$normalized)) == 1){
        choice <- good.moves$position
      } else {
        choice <- sample(good.moves$position, 1, prob=good.moves$normalized)
      }
      move.choice <- rep(0,64)
      move.choice[choice] <- 1
      if(player == "W"){
        moves.W[i.W,] <- move.choice
      } else {
        moves.B[i.B,] <- move.choice
      }
      ## Translate the choice back to the format of the game engine
      if(choice %% 8 == 0){
        move.y <- choice/8
        move.x <- 8
      } else {
        move.y <- (choice %/% 8) + 1
        move.x <- choice %% 8
      }
      ## Then, play PLAYER's move to update the board
      board <- play.move(board, player, c(move.x,move.y))
      board.digit <- model.board(board, player)
      if(player == "W"){
        outputs.W[i.W,,,] <- board.digit
        player <- "B"
        i.W <- i.W+1
      } else {
        outputs.B[i.B,,,] <- board.digit
        player <- "W"
        i.B <- i.B+1
      }
  }
  board.digit <- model.board(board, player)
  results <- determine.winner(board.digit, player, inputs.W, inputs.B,
                              moves.W, moves.B, outputs.W, outputs.B)
  return(results)
}

## create.training.set simulates n.games and compiles the resulting data into 
## the dataframes that will be used to train OthelloBot.
## is.it.na is a means of replacing the NaNs in the dataset with 0s in case
## the game takes less moves than the maximum to complete

create.training.set = function(n.games){
  winning.inputs <- array(data = NA, dim = c((n.games*30),8,8,1))
  losing.inputs <- array(data = NA, dim = c((n.games*30),8,8,1))
  winning.moves <- array(data = NA, dim = c((n.games*30), 64))
  losing.moves <- array(data = NA, dim = c((n.games*30), 64))
  winning.outputs <- array(data = NA, dim = c((n.games*30),8,8,1))
  losing.outputs <- array(data = NA, dim = c((n.games*30),8,8,1))
  for(i in 1:n.games){
    game.output <- play.game()
    winning.inputs[(1+(i-1)*30):(i*30),,,] <- game.output[[1]]
    losing.inputs[(1+(i-1)*30):(i*30),,,] <- game.output[[2]]
    winning.moves[(1+(i-1)*30):(i*30),] <- game.output[[3]]
    losing.moves[(1+(i-1)*30):(i*30),] <- game.output[[4]]
    winning.outputs[(1+(i-1)*30):(i*30),,,] <- game.output[[5]]
    losing.outputs[(1+(i-1)*30):(i*30),,,] <- game.output[[6]]
  }
  return(list(winning.inputs, losing.inputs, winning.moves, losing.moves,
              winning.outputs, losing.outputs))
}


#### 4) CONSTRUCTING THE CNN ####

## The code below initializes the model and defines its organization. 

model <- keras_model_sequential()

model %>%
  layer_conv_2d(filters=10, kernel_size=c(2,2), activation='relu', input_shape=c(8,8,1)) %>%
  layer_max_pooling_2d() %>%
  layer_flatten() %>%
  layer_dense(units=256, activation = 'relu') %>%
  layer_dense(units=64, activation='sigmoid')

summary(model)

model %>% compile(
  optimizer = 'adam',
  loss = 'mean_squared_error'
)

#### 5) RUNNING THE MODEL ####

## Lastly, this section runs the whole model. n.rounds determines how many
## cycles of training are to be run, while the for loop actually executes 
## the process. 

n.rounds <- 10
counter <- 1
boards.list <- list()
moves.list <- list()

for(i in 1:n.rounds){
  trainer <- create.training.set(50)
  good_board_init <- trainer[[1]]
  good_moves <- trainer[[3]]
  noNAs <- !is.na(good_moves[,1])
  good_board_init_clean <- good_board_init[noNAs,,,]
  good_board_init_clean <-
    array_reshape(good_board_init_clean, dim=c(dim(good_board_init_clean),1))
  good_moves_clean <- good_moves[noNAs,]
  bad_board_init <- trainer[[2]]
  bad_moves <- trainer[[4]]
  noNAs.bad <- !is.na(bad_moves[,1])
  bad_board_init_clean <- bad_board_init[noNAs.bad,,,]
  bad_board_init_clean <-
    array_reshape(bad_board_init_clean, dim=c(dim(bad_board_init_clean),1))
  bad_moves_clean <- bad_moves[noNAs.bad,]
  model %>% fit(good_board_init_clean, good_moves_clean, epochs=100)
  model %>% fit(bad_board_init_clean, bad_moves_clean, epochs=100)
  boards.list[[counter]] <- good_board_init_clean
  moves.list[[counter]] <- good_moves_clean
  counter <- counter + 1
}

#### 6) ANALYSIS SCRIPT ####

## Here are the analyses specifically targeting investigation of the common edge
## and corner strategy in playing Othello. First, the data is split up by the 
## ten rounds of training

boards.1 <- boards.list[[1]]
boards.2 <- boards.list[[2]]
boards.3 <- boards.list[[3]]
boards.4 <- boards.list[[4]]
boards.5 <- boards.list[[5]]
boards.6 <- boards.list[[6]]
boards.7 <- boards.list[[7]]
boards.8 <- boards.list[[8]]
boards.9 <- boards.list[[9]]
boards.10 <- boards.list[[10]]

moves.1 <- moves.list[[1]]
moves.2 <- moves.list[[2]]
moves.3 <- moves.list[[3]]
moves.4 <- moves.list[[4]]
moves.5 <- moves.list[[5]]
moves.6 <- moves.list[[6]]
moves.7 <- moves.list[[7]]
moves.8 <- moves.list[[8]]
moves.9 <- moves.list[[9]]
moves.10 <- moves.list[[10]]

## game.starts returns which turns were also the starts of specific games. This
## is important to determine number of turns until the corners were played.

game.starts = function(boardset){
  start.moves <- array(1)
  counter <- 2
  for(i in 2:(length(boardset)/64)){
    zero.count <- sum(boardset[i,,,]==0)
    zero.count.prev <- sum(boardset[i-1,,,]==0)
    if(zero.count > zero.count.prev){
      start.moves <- append(start.moves, counter)
      counter <- counter + 1
    } else {
      counter <- counter + 1
    }
  }
  return(start.moves)
}

## corners.analysis returns which turns the model played a move in one of the 
## four corners of the board. 

corners.analysis = function(moveset){
  top.left.moves <- array()
  top.right.moves <- array()
  bottom.left.moves <- array()
  bottom.right.moves <- array()
  counter <- 1
  for(i in 1:(length(moveset)/64)){
    move <- moveset[i,]
    if(move[1] == 1){
      top.left.moves <- append(top.left.moves, counter)
      counter <- counter + 1
    } else {
      if(move[8] == 1){
        top.right.moves <- append(top.right.moves, counter)
        counter <- counter + 1
      } else {
        if(move[57] == 1){
          bottom.left.moves <- append(bottom.left.moves, counter)
          counter <- counter + 1
        } else {
          if(move[64] == 1){
            bottom.right.moves <- append(bottom.right.moves, counter)
            counter <- counter + 1
          } else {
            counter <- counter + 1
          }
        }
      }
    }
  }
  noNAs.TL <- !is.na(top.left.moves)
  top.left.moves <- top.left.moves[noNAs.TL]
  noNAs.TR <- !is.na(top.right.moves)
  top.right.moves <- top.right.moves[noNAs.TR]
  noNAs.BL <- !is.na(bottom.left.moves)
  bottom.left.moves <- bottom.left.moves[noNAs.BL]
  noNAs.BR <- !is.na(bottom.right.moves)
  bottom.right.moves <- bottom.right.moves[noNAs.BR]
  return(list(top.left.moves, top.right.moves, 
              bottom.left.moves, bottom.right.moves))
}

get.rounds.no.play <- function(x, y, data){
  data <- as.data.frame(data)
  tmp <- data %>%
    filter(data <= y, x < data)
  return(length(tmp$data))
}

add.nas = function(length.no.move, corner, x){
  x <- x-1
  if(!length.no.move==0){
    for(i in 1:length.no.move){
      corner <- append(corner, NA, x)
    }
    } else {
      corner <- corner
    }
  return(corner)
}

process.corner = function(corner, game.starts){
  corner <- corner[[1]]
  corner.length <- length(corner)
  final.no.play <- get.rounds.no.play(corner[corner.length], 1486, game.starts)
  if(!final.no.play==0){
    for(i in 1:final.no.play){
      corner <- append(corner, NA, (length(corner)+1))
    }
  }else{
    corner <- corner
  }
  
  initial.no.play <- get.rounds.no.play(1, corner[1], game.starts)
  if(!initial.no.play==0){
    for(i in 1:initial.no.play){
      corner <- append(corner, NA, 0)
    }
  }else{
    corner <- corner
  }
  
  for(i in 2:50){
    if(!is.na(corner[i]) & !is.na(corner[i-1])){
      length.no.move <- get.rounds.no.play(corner[i-1], corner[i], game.starts)
      ## Take out one so it doesn't reuse from the last calculation
      length.no.move <- length.no.move - 1
      corner <- add.nas(length.no.move, corner, i)
      corner.length <- length(corner)
    } else {
      corner <- corner
    }
  }
  
  return(corner)
}

make.data.set = function(moveset, boardset){
  game.starts <- game.starts(boardset)
  four.corners <- corners.analysis(moveset)
  top.left <- process.corner(four.corners[1], game.starts)
  top.right <- process.corner(four.corners[2], game.starts)
  bottom.left <- process.corner(four.corners[3], game.starts)
  bottom.right <- process.corner(four.corners[4], game.starts)
  df <- data.frame(matrix(ncol=5, nrow=50))
  colnames(df) <-c('game.starts', 'top.left', 'top.right', 
                   'bottom.left', 'bottom.right')
  df$game.starts <- game.starts
  df$top.left <- top.left
  df$top.right <- top.right
  df$bottom.left <- bottom.left
  df$bottom.right <- bottom.right
  return(df)
}

## MAKE THE DATA SETS HERE ##
df1 <- make.data.set(moves.1, boards.1)
df2 <- make.data.set(moves.2, boards.2)
df3 <- make.data.set(moves.3, boards.3)
df4 <- make.data.set(moves.4, boards.4)
df5 <- make.data.set(moves.5, boards.5)
df6 <- make.data.set(moves.6, boards.6)
df7 <- make.data.set(moves.7, boards.7)
df8 <- make.data.set(moves.8, boards.8)
df9 <- make.data.set(moves.9, boards.9)
df10 <- make.data.set(moves.10, boards.10)

## Apologies that after this is wildly inefficient, I tried making the corner a 
## variable and after wrestling with everything else for so long this ended up 
## being easier for me somehow...

## STATS ##

calculate.turn.avg.TL = function(dataframe){
  dataframe <- dataframe %>% 
    filter(!is.na(top.left)) %>% 
    select(game.starts, top.left) %>% 
    mutate(turns.to.play = top.left - game.starts)
  mean <- mean(dataframe$turns.to.play)
  return(mean)
}

calculate.turn.avg.TR = function(dataframe){
  dataframe <- dataframe %>% 
    filter(!is.na(top.right)) %>% 
    select(game.starts, top.right) %>% 
    mutate(turns.to.play = top.right - game.starts)
  mean <- mean(dataframe$turns.to.play)
  return(mean)
}

calculate.turn.avg.BL = function(dataframe){
  dataframe <- dataframe %>% 
    filter(!is.na(bottom.left)) %>% 
    select(game.starts, bottom.left) %>% 
    mutate(turns.to.play = bottom.left - game.starts)
  mean <- mean(dataframe$turns.to.play)
  return(mean)
}

calculate.turn.avg.BR = function(dataframe){
  dataframe <- dataframe %>% 
    filter(!is.na(bottom.right)) %>% 
    select(game.starts, bottom.right) %>% 
    mutate(turns.to.play = bottom.right - game.starts)
  mean <- mean(dataframe$turns.to.play)
  return(mean)
}

calculate.length.TL = function(dataframe){
  dataframe <- dataframe %>% 
    filter(!is.na(top.left))
  length <- length(dataframe$game.starts)
  return(length)
}

calculate.length.TR = function(dataframe){
  dataframe <- dataframe %>% 
    filter(!is.na(top.right))
  length <- length(dataframe$game.starts)
  return(length)
}

calculate.length.BL = function(dataframe){
  dataframe <- dataframe %>% 
    filter(!is.na(bottom.left))
  length <- length(dataframe$game.starts)
  return(length)
}

calculate.length.BR = function(dataframe){
  dataframe <- dataframe %>% 
    filter(!is.na(bottom.right))
  length <- length(dataframe$game.starts)
  return(length)
}

overall.stats.df <- data.frame(matrix(ncol=8, nrow=10))
colnames(overall.df) <-c('mean.TL', 'n.TL', 'mean.TR', 'n.TR', 
                 'mean.BL', 'n.BL', 'mean.BR', 'n.BR')
overall.stats.df$mean.TL <- c(calculate.turn.avg.TL(df1),
                        calculate.turn.avg.TL(df2),
                        calculate.turn.avg.TL(df3),
                        calculate.turn.avg.TL(df4),
                        calculate.turn.avg.TL(df5),
                        calculate.turn.avg.TL(df6),
                        calculate.turn.avg.TL(df7),
                        calculate.turn.avg.TL(df8),
                        calculate.turn.avg.TL(df9),
                        calculate.turn.avg.TL(df10))
overall.stats.df$mean.TR <- c(calculate.turn.avg.TR(df1),
                        calculate.turn.avg.TR(df2),
                        calculate.turn.avg.TR(df3),
                        calculate.turn.avg.TR(df4),
                        calculate.turn.avg.TR(df5),
                        calculate.turn.avg.TR(df6),
                        calculate.turn.avg.TR(df7),
                        calculate.turn.avg.TR(df8),
                        calculate.turn.avg.TR(df9),
                        calculate.turn.avg.TR(df10))
overall.stats.df$mean.BL <- c(calculate.turn.avg.BL(df1),
                        calculate.turn.avg.BL(df2),
                        calculate.turn.avg.BL(df3),
                        calculate.turn.avg.BL(df4),
                        calculate.turn.avg.BL(df5),
                        calculate.turn.avg.BL(df6),
                        calculate.turn.avg.BL(df7),
                        calculate.turn.avg.BL(df8),
                        calculate.turn.avg.BL(df9),
                        calculate.turn.avg.BL(df10))
overall.stats.df$mean.BR <- c(calculate.turn.avg.BR(df1),
                        calculate.turn.avg.BR(df2),
                        calculate.turn.avg.BR(df3),
                        calculate.turn.avg.BR(df4),
                        calculate.turn.avg.BR(df5),
                        calculate.turn.avg.BR(df6),
                        calculate.turn.avg.BR(df7),
                        calculate.turn.avg.BR(df8),
                        calculate.turn.avg.BR(df9),
                        calculate.turn.avg.BR(df10))
overall.stats.df$n.TL <- c(calculate.length.TL(df1),
                        calculate.length.TL(df2),
                        calculate.length.TL(df3),
                        calculate.length.TL(df4),
                        calculate.length.TL(df5),
                        calculate.length.TL(df6),
                        calculate.length.TL(df7),
                        calculate.length.TL(df8),
                        calculate.length.TL(df9),
                        calculate.length.TL(df10))
overall.stats.df$n.TR <- c(calculate.length.TR(df1),
                        calculate.length.TR(df2),
                        calculate.length.TR(df3),
                        calculate.length.TR(df4),
                        calculate.length.TR(df5),
                        calculate.length.TR(df6),
                        calculate.length.TR(df7),
                        calculate.length.TR(df8),
                        calculate.length.TR(df9),
                        calculate.length.TR(df10))
overall.stats.df$n.BL <- c(calculate.length.BL(df1),
                        calculate.length.BL(df2),
                        calculate.length.BL(df3),
                        calculate.length.BL(df4),
                        calculate.length.BL(df5),
                        calculate.length.BL(df6),
                        calculate.length.BL(df7),
                        calculate.length.BL(df8),
                        calculate.length.BL(df9),
                        calculate.length.BL(df10))
overall.stats.df$n.BR <- c(calculate.length.BR(df1),
                        calculate.length.BR(df2),
                        calculate.length.BR(df3),
                        calculate.length.BR(df4),
                        calculate.length.BR(df5),
                        calculate.length.BR(df6),
                        calculate.length.BR(df7),
                        calculate.length.BR(df8),
                        calculate.length.BR(df9),
                        calculate.length.BR(df10))

## BOXPLOTS ##

calculate.boxplot.TL = function(dataframe){
  dataframe <- dataframe %>% 
    filter(!is.na(top.left)) %>% 
    select(game.starts, top.left) %>% 
    mutate(turns.to.play = top.left - game.starts) %>% 
    mutate(corner = "Top Left") %>% 
    rename(turns.until.corner = top.left)
}
calculate.boxplot.TR = function(dataframe){
  dataframe <- dataframe %>% 
    filter(!is.na(top.right)) %>% 
    select(game.starts, top.right) %>% 
    mutate(turns.to.play = top.right - game.starts) %>% 
    mutate(corner = "Top Right") %>% 
    rename(turns.until.corner = top.right)
}
calculate.boxplot.BL = function(dataframe){
  dataframe <- dataframe %>% 
    filter(!is.na(bottom.left)) %>% 
    select(game.starts, bottom.left) %>% 
    mutate(turns.to.play = bottom.left - game.starts) %>% 
    mutate(corner = "Bottom Left") %>% 
    rename(turns.until.corner = bottom.left)
}
calculate.boxplot.BR = function(dataframe){
  dataframe <- dataframe %>% 
    filter(!is.na(bottom.right)) %>% 
    select(game.starts, bottom.right) %>% 
    mutate(turns.to.play = bottom.right - game.starts) %>% 
    mutate(corner = "Bottom Right") %>% 
    rename(turns.until.corner = bottom.right)
}

boxplot1.df <- rbind(calculate.boxplot.TL(df1), calculate.boxplot.TR(df1),
                       calculate.boxplot.BL(df1), calculate.boxplot.BR(df1))
boxplot1.df$round <- "No Training"

boxplot2.df <- rbind(calculate.boxplot.TL(df2), calculate.boxplot.TR(df2),
                     calculate.boxplot.BL(df2), calculate.boxplot.BR(df2))
boxplot2.df <- boxplot2.df %>% 
  mutate(round = "First Training")

boxplot3.df <- rbind(calculate.boxplot.TL(df3), calculate.boxplot.TR(df3),
                     calculate.boxplot.BL(df3), calculate.boxplot.BR(df3))
boxplot3.df <- boxplot3.df %>% 
  mutate(round = "Second Training")

boxplot4.df <- rbind(calculate.boxplot.TL(df4), calculate.boxplot.TR(df4),
                     calculate.boxplot.BL(df4), calculate.boxplot.BR(df4))
boxplot4.df <- boxplot4.df %>% 
  mutate(round = "Third Training")

boxplot5.df <- rbind(calculate.boxplot.TL(df5), calculate.boxplot.TR(df5),
                     calculate.boxplot.BL(df5), calculate.boxplot.BR(df5))
boxplot5.df <- boxplot5.df %>% 
  mutate(round = "Fourth Training")

boxplot.df <- rbind(boxplot1.df, boxplot2.df, boxplot3.df,
                    boxplot4.df, boxplot5.df)

## PRINT BOXPLOTS HERE ##

boxplot.df$round <- factor(boxplot.df$round, 
                           levels=c("No Training", "First Training", 
                                    "Second Training", "Third Training",
                                    "Fourth Training"))

boxplot(turns.to.play ~ round, boxplot.df)

## CORNERS ACQUIRED ##

calculate.proportion = function(df){
  TL.count <- table(!is.na(df$top.left))["TRUE"]
  TR.count <- table(!is.na(df$top.right))["TRUE"]
  BL.count <- table(!is.na(df$bottom.left))["TRUE"]
  BR.count <- table(!is.na(df$bottom.right))["TRUE"]
  TL.prop <- TL.count/50
  TR.prop <- TR.count/50
  BL.prop <- BL.count/50
  BR.prop <- BR.count/50
  return(c(TL.prop, TR.prop, BL.prop, BR.prop))
}

proportions = data.frame(matrix(ncol=0, nrow=4))
rownames(proportions) <-c('TL.prop', 'TR.prop', 'BL.prop', 'BR.prop')
proportions$df1 <- calculate.proportion(df1)
proportions$df2 <- calculate.proportion(df2)
proportions$df3 <- calculate.proportion(df3)
proportions$df4 <- calculate.proportion(df4)
proportions$df5 <- calculate.proportion(df5)
proportions$df6 <- calculate.proportion(df6)
proportions$df7 <- calculate.proportion(df7)
proportions$df8 <- calculate.proportion(df8)
proportions$df9 <- calculate.proportion(df9)
proportions$df10 <- calculate.proportion(df10)

proportions2 <- data.frame(t(proportions))
proportions2 <- proportions2 %>% 
  mutate(combo.proportion = (TL.prop+TR.prop+BL.prop+BR.prop) / 4)

## MAKE GRAPH HERE ##

plot(proportions2$combo.proportion, 
     type = "b",
     main = "Proportion of Corners Acquired per Training Round",
     xlab="Training Round", 
     ylab="Proportion of Corners Acquired")

