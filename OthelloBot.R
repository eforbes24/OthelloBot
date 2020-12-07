## Eden Forbes
## OthelloBot
Sys.setenv("CUDA_VISIBLE_DEVICES"="1")

library(keras)
library(dplyr)

## GAME ENGINE
## Borrowed & Edited from Jack Davis, www.stats-et-al.com"
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

## CONSTRUCT CNN

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
  loss = 'rmse'
)

## Q-LEARNING W/ CNN FUNCTIONS

## Note that it reads columns instead of rows, so the ninth position is the
## first row of the second column

## Give the legal moves for the player given the board
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

## Translate the game engine board format to the CNN board format
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

## Play a full game of Othello
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
      print(board)
  }
  board.digit <- model.board(board, player)
  results <- determine.winner(board.digit, player, inputs.W, inputs.B,
                              moves.W, moves.B, outputs.W, outputs.B)
  return(results)
}


## NEED TO FIX DATA FORMATTING
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

## RUNNING THE MODEL

trainer <- create.training.set(1)
board_init <- trainer[[1]]
moves <- trainer[[3]]

model %>% fit(board_init, moves, epochs=50)








