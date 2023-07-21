# Author: Sergej Levich
# Journal article: Sergej Levich et al., Decision Support Systems, https://doi.org/10.1016/j.dss.2023.114043

suppressMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(magrittr)
  library(ggplot2)
})
theme_set(theme_bw())
options(pillar.sigfig = 3)

# evaluate models for type target
## load evaluation dataframe
evaluation <- tibble()
for (i in list.files('results/evaluation/val_set/type/')) {
  evaluation %<>% 
    bind_rows(
      read_csv(paste0('results/evaluation/val_set/type/', i), show_col_types = FALSE)
    )
}
evaluation %<>% 
  mutate(
    context = factor(context, levels = unique(context)),
    data_split = factor(data_split, levels = unique(data_split)),
    split_type = ifelse(fold %in% (0:9), 'cross_validation', 'rolling_window')
  ) %>% 
  select(grid_row, context, split_type, data_split, fold, best_epoch, everything()) %>% 
  filter(data_split == 'val_set')

## load training history
history <- tibble()
for (i in list.files('results/training_history/type/')) {
  history %<>% 
    bind_rows(
      read_csv(paste0('results/training_history/type/', i), show_col_types = FALSE) %>% 
        mutate(epoch = 1:n())
    )
}
history %<>%
  mutate(split_type = ifelse(fold %in% (0:9), 'cross_validation', 'rolling_window'))

## show training history
history %>% 
  pivot_longer(c(loss, val_loss), 'key', 'value') %>% 
  filter(key == 'val_loss') %>% 
  ggplot(aes(epoch, value, color = as.factor(fold))) +
  geom_line() +
  facet_wrap(grid_row ~ context, nrow = 8, scales = 'free_y') + 
  theme(legend.position = 'none')

## show losses per iteration
evaluation %>%
  select(context, split_type, data_split, total_loss, fold, grid_row) %>%
  ggplot(aes(fold, total_loss, fill = split_type)) +
  geom_col() +
  facet_wrap(grid_row ~ context, nrow = 8)

## calculate mean values of performance metrics
evaluation %>% 
  group_by(context, fold) %>% 
  filter(total_loss == min(total_loss)) %>% 
  group_by(split_type, data_split, context) %>% 
  summarise(
    n_models = n(),
    n_folds = n_distinct(fold),
    n_grid_rows = n_distinct(grid_row),
    target = unique(target),
    mean_best_epoch = mean(best_epoch),
    loss_mean = mean(total_loss),
    loss_sd = sd(total_loss),
    type_accuracy_mean = mean(type_cls_accuracy),
    type_accuracy_sd = sd(type_cls_accuracy),
    .groups = 'drop'
  ) %>% 
  arrange(split_type, -loss_mean)

## check performance metrics calculation
read_csv('results/evaluation/test_set/type/evaluation.csv', show_col_types = FALSE) %>%
  select(context, fold, type_cls_accuracy) %>% 
  left_join(
    read_csv('results/evaluation/test_set/type/predictions.csv', show_col_types = FALSE) %>% 
      group_by(context, fold) %>% 
      summarise(accuracy = sum(type_cls_actuals == type_cls_prediction) / n()) %>% 
      ungroup(),
    by = c('context', 'fold')
  ) %>% 
  summarise(check_accuracy = sum(type_cls_accuracy - accuracy))

## show performance of best configuration per fold
read_csv('results/evaluation/test_set/type/evaluation.csv', show_col_types = FALSE) %>%
  mutate(split_type = ifelse(fold %in% (0:9), 'cross_validation', 'rolling_window')) %>% 
  ggplot(aes(fold, type_cls_accuracy, fill = split_type)) +
  geom_col() +
  facet_wrap( ~ forcats::fct_rev(context), nrow = 1)

## show mean performance of best configurations on test set
read_csv('results/evaluation/test_set/type/evaluation.csv', show_col_types = FALSE) %>%
  mutate(split_type = ifelse(fold %in% (0:9), 'cross_validation', 'rolling_window')) %>% 
  group_by(split_type, data_split, context) %>% 
  summarise(
    n_models = n(),
    n_folds = n_distinct(fold),
    target = unique(target),
    loss_mean = mean(total_loss),
    loss_sd = sd(total_loss),
    type_accuracy_mean = mean(type_cls_accuracy),
    type_accuracy_sd = sd(type_cls_accuracy),
    .groups = 'drop'
  ) %>% 
  arrange(split_type, -loss_mean)

# evaluate models for event and time targets
## load evaluation dataframe
evaluation <- tibble()
for (i in list.files('results/evaluation/val_set/event_time/')) {
  evaluation %<>% 
    bind_rows(
      read_csv(paste0('results/evaluation/val_set/event_time/', i), show_col_types = FALSE)
    )
}
evaluation %<>% 
  mutate(
    context = factor(context, levels = unique(context)),
    data_split = factor(data_split, levels = unique(data_split)),
    split_type = ifelse(fold %in% (0:9), 'cross_validation', 'rolling_window')
  ) %>% 
  select(grid_row, context, split_type, data_split, fold, best_epoch, everything()) %>% 
  filter(data_split == 'val_set')

## load training history
history <- tibble()
for (i in list.files('results/training_history/event_time/')) {
  history %<>% 
    bind_rows(
      read_csv(paste0('results/training_history/event_time/', i), show_col_types = FALSE) %>% 
        mutate(epoch = 1:n())
    )
}
history %<>%
  mutate(split_type = ifelse(fold %in% (0:9), 'cross_validation', 'rolling_window'))

## show training history
history %>% 
  pivot_longer(c(loss, val_loss), 'key', 'value') %>% 
  filter(key == 'val_loss') %>% 
  ggplot(aes(epoch, value, color = as.factor(fold))) +
  geom_line() +
  facet_wrap(grid_row ~ context, nrow = 8, scales = 'free_y') + 
  theme(legend.position = 'none')

## show losses per iteration
evaluation %>%
  select(context, split_type, data_split, total_loss, fold, grid_row) %>%
  ggplot(aes(fold, total_loss, fill = split_type)) +
  geom_col() +
  facet_wrap(grid_row ~ context, nrow = 8)

## calculate mean values of performance metrics
evaluation %>% 
  group_by(context, fold) %>% 
  filter(total_loss == min(total_loss)) %>% 
  group_by(split_type, data_split, context) %>% 
  summarise(
    n_models = n(),
    n_folds = n_distinct(fold),
    target = unique(target),
    mean_best_epoch = mean(best_epoch),
    loss_mean = mean(total_loss),
    loss_sd = sd(total_loss),
    event_accuracy_mean = mean(event_cls_accuracy),
    event_accuracy_sd = sd(event_cls_accuracy),
    time_mse_mean = mean(time_reg_mse),
    time_mse_sd = sd(time_reg_mse),
    .groups = 'drop'
  ) %>% 
  arrange(split_type, -loss_mean)

## check performance metrics calculation
read_csv('results/evaluation/test_set/event_time/evaluation.csv', show_col_types = FALSE) %>%
  select(context, fold, event_cls_accuracy, time_reg_mse) %>% 
  left_join(
    read_csv('results/evaluation/test_set/event_time/predictions.csv', show_col_types = FALSE) %>% 
      group_by(context, fold) %>% 
      summarise(
        accuracy = sum(event_cls_actuals == event_cls_prediction) / n(),
        mse = mean((time_reg_actuals - time_reg_prediction)^2)
      ) %>% 
      ungroup(),
    by = c('context', 'fold')
  ) %>% 
  summarise(
    check_accuracy = sum(event_cls_accuracy - accuracy),
    check_mse = sum(time_reg_mse - mse)
  )

## show performance of best configuration per fold
read_csv('results/evaluation/test_set/event_time/evaluation.csv', show_col_types = FALSE) %>%
  mutate(split_type = ifelse(fold %in% (0:9), 'cross_validation', 'rolling_window')) %>% 
  ggplot(aes(fold, event_cls_accuracy, fill = split_type)) +
  geom_col() +
  facet_wrap( ~ forcats::fct_rev(context), nrow = 1)

read_csv('results/evaluation/test_set/event_time/evaluation.csv', show_col_types = FALSE) %>%
  mutate(split_type = ifelse(fold %in% (0:9), 'cross_validation', 'rolling_window')) %>% 
  ggplot(aes(fold, time_reg_mse, fill = split_type)) +
  geom_col() +
  facet_wrap( ~ forcats::fct_rev(context), nrow = 1)

## show mean performance of best configurations on test set
read_csv('results/evaluation/test_set/event_time/evaluation.csv', show_col_types = FALSE) %>%
  mutate(split_type = ifelse(fold %in% (0:9), 'cross_validation', 'rolling_window')) %>% 
  group_by(split_type, data_split, context) %>% 
  summarise(
    n_models = n(),
    n_folds = n_distinct(fold),
    target = unique(target),
    loss_mean = mean(total_loss),
    loss_sd = sd(total_loss),
    event_accuracy_mean = mean(event_cls_accuracy),
    event_accuracy_sd = sd(event_cls_accuracy),
    time_mse_mean = mean(time_reg_mse),
    time_mse_sd = sd(time_reg_mse),
    .groups = 'drop'
  ) %>% 
  arrange(split_type, -loss_mean)
