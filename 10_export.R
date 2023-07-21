# Author: Sergej Levich
# Journal article: Sergej Levich et al., Decision Support Systems, https://doi.org/10.1016/j.dss.2023.114043

suppressMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(purrr)
  library(magrittr)
  library(ggplot2)
  library(caret)
  library(bupaR)
})
theme_set(theme_bw())
options(pillar.sigfig = 3)

# create directory for export
dir.create('results/export')

# load log data
log_df <- read_csv('log_data/process_log.csv', show_col_types = FALSE)

# table 4
read_csv('results/evaluation/test_set/type/evaluation.csv', show_col_types = FALSE) %>% 
  mutate(split_type = ifelse(fold %in% (0:9), 'cross_validation', 'rolling_window')) %>% 
  filter(split_type == 'cross_validation') %>% 
  group_by(context) %>% 
  summarise(
    type_cls_accuracy_mean = mean(type_cls_accuracy),
    type_cls_accuracy_sd = sd(type_cls_accuracy)
  ) %>% 
  full_join(
    read_csv('results/evaluation/test_set/event_time/evaluation.csv', show_col_types = FALSE) %>% 
      mutate(split_type = ifelse(fold %in% (0:9), 'cross_validation', 'rolling_window')) %>% 
      filter(split_type == 'cross_validation') %>% 
      group_by(context) %>% 
      summarise(
        event_cls_accuracy_mean = mean(event_cls_accuracy),
        event_cls_accuracy_sd = sd(event_cls_accuracy),
        time_reg_mse_mean = mean(time_reg_mse),
        time_reg_mse_sd = sd(time_reg_mse)
      ),
    by = 'context'
  ) %>% 
  write_csv('results/export/table_4.csv')

# load predictions for target type
type_predictions <- read_csv('results/evaluation/test_set/type/predictions.csv', show_col_types = FALSE) %>% 
  mutate(split_type = ifelse(fold %in% (0:9), 'cross_validation', 'rolling_window')) %>% 
  group_by(context, fold, instance_id) %>% 
  mutate(event_nr = 1:n()) %>% 
  ungroup() %>% 
  left_join(
    log_df %>% 
      distinct(type, type_name) %>% 
      transmute(actuals = type_name, type),
    by = c('type_cls_actuals' = 'type')
  ) %>% 
  left_join(
    log_df %>% 
      distinct(type, type_name) %>% 
      transmute(prediction = type_name, type),
    by = c('type_cls_prediction' = 'type')
  ) %>% 
  left_join(
    log_df %>% 
      group_by(instance_id) %>% 
      mutate(event_nr = 1:n()) %>% 
      ungroup() %>% 
      select(instance_id, event_nr, file_name, n_pages, type_name, state_name, log_time_since_last_event),
    by = c('instance_id', 'event_nr')
  ) %>% 
  select(-type_cls_actuals, -type_cls_prediction) %>% 
  mutate(
    actuals = factor(actuals),
    prediction = factor(prediction, levels = levels(actuals)),
    context = factor(context, levels = unique(context))
  )

# table 5
type_predictions %>% 
  filter(split_type == 'cross_validation') %>% 
  filter(context %in% c('none', 'doc_features_bert_german')) %>% 
  group_by(context, fold) %>% 
  summarise(metrics = list(confusionMatrix(prediction, actuals)$byClass), .groups = 'drop') %>% 
  mutate(metrics = map(metrics, as_tibble, rownames = 'type')) %>% 
  unnest(metrics) %>% 
  janitor::clean_names() %>% 
  transmute(fold, context, type = str_remove(type, 'Class: '), precision, recall) %>% 
  pivot_longer(c(precision, recall), 'metric', 'value') %>% 
  pivot_wider(names_from = context, values_from = value) %>% 
  group_by(type, metric) %>% 
  summarise(
    NAs_none = sum(is.na(none)),
    NAs_doc_features_bert_german = sum(is.na(doc_features_bert_german)),
    mean_none = mean(none, na.rm = FALSE),
    sd_none = sd(none, na.rm = FALSE),
    mean_doc_features_bert_german = mean(doc_features_bert_german, na.rm = FALSE),
    sd_doc_features_bert_german = sd(doc_features_bert_german),
    pcnt_improvement = ((mean_doc_features_bert_german / mean_none) - 1) * 100,
    p_value = ifelse(any(is.na(none)) | any(is.na(doc_features_bert_german)), NA,
                     wilcox.test(none, doc_features_bert_german, alternative = 'two.sided', paired = FALSE, exact = FALSE)$p.value),
    p_star = ifelse(!is.na(p_value), ifelse(p_value < 0.05, ifelse(p_value < 0.001, '***', ifelse(p_value < 0.01, '**', '*')), ''), NA)
  ) %>% 
  ungroup() %>% 
  left_join(
    type_predictions %>% 
      distinct(instance_id, event_nr, actuals) %>% 
      count(actuals) %>% 
      transmute(type = actuals, prevalence = n / sum(n)), # prevalence is calculated based on number of instances
    by = 'type'
  ) %>% 
  arrange(metric, -prevalence) %>% 
  write_csv('results/export/table_5.csv')

# figures 6 & 7
type_predictions %>%
  filter(split_type == 'cross_validation') %>% 
  filter(context %in% c('none', 'doc_features_bert_german')) %>% 
  group_by(context, fold, instance_id) %>% 
  mutate(
    event_nr = 1:n(),
    correct = actuals == prediction,
    instance_length = n()
  ) %>% 
  group_by(context, instance_length, event_nr) %>% 
  summarise(pcnt_correct = sum(correct) / n(), n = n()) %>% 
  ungroup() %>% 
  mutate(context = factor(context), event_nr = factor(event_nr), instance_length = factor(instance_length)) %>% 
  pivot_wider(names_from = event_nr, values_from = pcnt_correct) %>% 
  write_csv('results/export/figures_6_7.csv')
  
# load predictions for target event & time
event_time_predictions <- read_csv('results/evaluation/test_set/event_time/predictions.csv', show_col_types = FALSE) %>% 
  mutate(split_type = ifelse(fold %in% (0:9), 'cross_validation', 'rolling_window')) %>% 
  group_by(context, fold, instance_id) %>% 
  mutate(event_nr = 1:n()) %>% 
  ungroup() %>% 
  left_join(
    log_df %>% 
      distinct(state, state_name) %>% 
      transmute(actuals = state_name, state) %>% 
      add_row(state = n_distinct(log_df$state), actuals = 'EOS'), # add EOS event
    by = c('event_cls_actuals' = 'state')
  ) %>% 
  left_join(
    log_df %>% 
      distinct(state, state_name) %>% 
      transmute(prediction = state_name, state) %>% 
      add_row(state = n_distinct(log_df$state), prediction = 'EOS'), # add EOS event
    by = c('event_cls_prediction' = 'state')
  ) %>% 
  left_join(
    log_df %>% 
      group_by(instance_id) %>% 
      mutate(event_nr = 1:n()) %>% 
      ungroup() %>% 
      select(instance_id, event_nr, file_name, n_pages, type_name, state_name, log_time_since_last_event),
    by = c('instance_id', 'event_nr')
  ) %>% 
  select(-event_cls_actuals, -event_cls_prediction) %>% 
  mutate(
    actuals = factor(actuals),
    prediction = factor(prediction, levels = levels(actuals)),
    context = factor(context, levels = unique(context))
  )

# table 6
event_time_predictions %>% 
  filter(split_type == 'cross_validation') %>% 
  filter(context %in% c('none', 'doc_features_bert_german')) %>% 
  group_by(context, fold) %>% 
  summarise(metrics = list(confusionMatrix(prediction, actuals)$byClass), .groups = 'drop') %>% 
  mutate(metrics = map(metrics, as_tibble, rownames = 'state')) %>% 
  unnest(metrics) %>% 
  janitor::clean_names() %>% 
  transmute(fold, context, state = str_remove(state, 'Class: '), precision, recall) %>% 
  pivot_longer(c(precision, recall), 'metric', 'value') %>% 
  pivot_wider(names_from = context, values_from = value) %>% 
  group_by(state, metric) %>% 
  summarise(
    NAs_none = sum(is.na(none)),
    NAs_doc_features_bert_german = sum(is.na(doc_features_bert_german)),
    mean_none = mean(none, na.rm = FALSE),
    sd_none = sd(none, na.rm = FALSE),
    mean_doc_features_bert_german = mean(doc_features_bert_german, na.rm = FALSE),
    sd_doc_features_bert_german = sd(doc_features_bert_german),
    pcnt_improvement = ((mean_doc_features_bert_german / mean_none) - 1) * 100,
    p_value = ifelse(any(is.na(none)) | any(is.na(doc_features_bert_german)), NA,
                     wilcox.test(none, doc_features_bert_german, alternative = 'two.sided', paired = FALSE, exact = FALSE)$p.value),
    p_star = ifelse(!is.na(p_value), ifelse(p_value < 0.05, ifelse(p_value < 0.001, '***', ifelse(p_value < 0.01, '**', '*')), ''), NA)
  ) %>% 
  ungroup() %>% 
  left_join(
    event_time_predictions %>% 
      distinct(instance_id, event_nr, actuals) %>% 
      count(actuals) %>% 
      transmute(state = actuals, prevalence = n / sum(n)), # prevalence is calculated based on number of single events
    by = 'state'
  ) %>% 
  arrange(metric, -prevalence) %>% 
  write_csv('results/export/table_6.csv')

# table 7
event_time_predictions %>% 
  filter(split_type == 'cross_validation') %>% 
  filter(context %in% c('none', 'doc_features_bert_german')) %>% 
  group_by(context, fold, actuals) %>% 
  summarise(mse = mean((time_reg_actuals - time_reg_prediction)^2), .groups = 'drop') %>% 
  pivot_wider(names_from = context, values_from = mse) %>% 
  group_by(next_event = actuals) %>% 
  summarise(
    mean_none = mean(none, na.rm = FALSE),
    sd_none = sd(none, na.rm = FALSE),
    mean_doc_features_bert_german = mean(doc_features_bert_german, na.rm = FALSE),
    sd_doc_features_bert_german = sd(doc_features_bert_german),
    pcnt_improvement = ((mean_doc_features_bert_german / mean_none) - 1) * 100,
    p_value = ifelse(any(is.na(none)) | any(is.na(doc_features_bert_german)), NA,
                     wilcox.test(none, doc_features_bert_german, alternative = 'two.sided', paired = FALSE, exact = FALSE)$p.value),
    p_star = ifelse(!is.na(p_value), ifelse(p_value < 0.05, ifelse(p_value < 0.001, '***', ifelse(p_value < 0.01, '**', '*')), ''), NA)
  ) %>% 
  left_join(
    event_time_predictions %>% 
      distinct(instance_id, event_nr, actuals) %>% 
      count(actuals) %>% 
      transmute(next_event = actuals, prevalence = n / sum(n)), # prevalence is calculated based on number of single events
    by = 'next_event'
  ) %>% 
  arrange(-prevalence) %>% 
  write_csv('results/export/table_7.csv')

# load shap values
shap_df <- bind_rows(
  read_csv('results/shap/doc_features_rvl/shap_df.csv', show_col_types = FALSE),
  read_csv('results/shap/doc_features_bert_german/shap_df.csv', show_col_types = FALSE)
)

# figures 8 & 9
shap_df %>% 
  filter(!is.na(file_name)) %>% 
  select(context, instance_id, state_name, type_name, file_name, n_pages, state_name, c(matches('[0-9]|[0-9][0-9]'))) %>% 
  pivot_longer(c(matches('[0-9]|[0-9][0-9]')), names_to = 'type', values_to = 'shap_value') %>% 
  left_join(
    log_df %>% 
      distinct(type, type_name) %>% 
      mutate(type = as.character(type)) %>% 
      rename(type_shap = type_name),
    by = 'type'
  ) %>% 
  group_by(context, state_name, type_shap) %>% 
  summarise(mean_abs_shap = mean(abs(shap_value))) %>% 
  ungroup() %>% 
  pivot_wider(names_from = state_name, values_from = mean_abs_shap) %>% 
  write_csv('results/export/figures_8_9.csv')

# table a3
read_csv('results/evaluation/test_set/type/evaluation.csv', show_col_types = FALSE) %>% 
  mutate(split_type = ifelse(fold %in% (0:9), 'cross_validation', 'rolling_window')) %>% 
  filter(split_type == 'cross_validation') %>% 
  mutate(fold = fold + 1) %>% 
  group_by(context, layer_size, learning_rate) %>% 
  summarise(fold = paste(fold, collapse = ',')) %>% 
  select(context, fold, layer_size, learning_rate) %>% 
  ungroup() %>% 
  write_csv('results/export/table_a3.csv')

# table a4
read_csv('results/evaluation/test_set/event_time/evaluation.csv', show_col_types = FALSE) %>% 
  mutate(split_type = ifelse(fold %in% (0:9), 'cross_validation', 'rolling_window')) %>% 
  filter(split_type == 'cross_validation') %>% 
  mutate(fold = fold + 1) %>% 
  group_by(context, layer_size, learning_rate) %>% 
  summarise(fold = paste(fold, collapse = ',')) %>% 
  select(context, fold, layer_size, learning_rate) %>% 
  ungroup() %>% 
  write_csv('results/export/table_a4.csv')

# table c1
log_df %>% 
  group_by(instance_id) %>% 
  transmute(current_state = state_name, next_state = lead(current_state, default = 'EOS')) %>% 
  ungroup() %>% 
  count(current_state, next_state) %>% 
  group_by(current_state) %>% 
  transmute(next_state, pcnt = n / sum(n)) %>% 
  spread(next_state, pcnt) %>% 
  write_csv('results/export/table_c1.csv')

# table c2
log_df %>% 
  mutate(event_id = 1:n(), status = 'complete', resource = NA) %>% 
  eventlog(
    case_id = 'instance_id',
    activity_id = 'state_name',
    activity_instance_id = 'event_id',
    lifecycle_id = 'status',
    timestamp = 'timestamp',
    resource_id = 'resource'
  ) %>% 
  trace_coverage('trace') %>% 
  head(10) %>% 
  write_csv('results/export/table_c2.csv')

# figure d1
event_time_predictions %>% 
  filter(split_type == 'cross_validation') %>%
  filter(context %in% c('none', 'doc_features_bert_german')) %>% 
  left_join(
    log_df %>% 
      group_by(instance_id) %>% 
      transmute(instance_id, event_nr = 1:n(), state_name) %>% 
      ungroup(),
    by = c('instance_id', 'event_nr')
  ) %>% 
  group_by(context, fold, actuals) %>% 
  summarise(mse = mean((time_reg_actuals - time_reg_prediction)^2), .groups = 'drop') %>% 
  group_by(context, actuals) %>% 
  summarise(boxplot = list(as_tibble(t(boxplot(mse, plot = FALSE)$stats)))) %>% 
  unnest(boxplot) %>% 
  rename(lower_whisker = V1, q1 = V2, median = V3, q3 = V4, upper_whisker = V5) %>% 
  write_csv('results/export/figure_d1.csv')

# table e1
read_csv('results/evaluation/test_set/type/evaluation.csv', show_col_types = FALSE) %>% 
  mutate(split_type = ifelse(fold %in% (0:9), 'cross_validation', 'rolling_window')) %>% 
  filter(split_type == 'rolling_window') %>% 
  group_by(context) %>% 
  summarise(
    type_cls_accuracy_mean = mean(type_cls_accuracy),
    type_cls_accuracy_sd = sd(type_cls_accuracy)
  ) %>% 
  full_join(
    read_csv('results/evaluation/test_set/event_time/evaluation.csv', show_col_types = FALSE) %>% 
      mutate(split_type = ifelse(fold %in% (0:9), 'cross_validation', 'rolling_window')) %>% 
      filter(split_type == 'rolling_window') %>% 
      group_by(context) %>% 
      summarise(
        event_cls_accuracy_mean = mean(event_cls_accuracy),
        event_cls_accuracy_sd = sd(event_cls_accuracy),
        time_reg_mse_mean = mean(time_reg_mse),
        time_reg_mse_sd = sd(time_reg_mse)
      ),
    by = 'context'
  ) %>% 
  write_csv('results/export/table_e1.csv')

# figures f1-f6
shap_df %>% 
  filter(instance_id %in% c(32, 986, 4252)) %>% 
  group_by(context, instance_id) %>% 
  mutate(event_name = paste(1:n(), state_name, sep = '. ')) %>%
  ungroup() %>% 
  select(instance_id, context, event_name, file_name, c(matches('[0-9]|[0-9][0-9]'))) %>% 
  pivot_longer(c(matches('[0-9]|[0-9][0-9]')), names_to = 'type', values_to = 'shap_value') %>% 
  left_join(
    log_df %>% 
      distinct(type, type_name) %>% 
      mutate(type = as.character(type)) %>% 
      rename(type_shap = type_name),
    by = 'type'
  ) %>% 
  mutate(shap_value = ifelse(is.na(file_name), NA, shap_value)) %>% 
  left_join(
    type_predictions %>% 
      filter(split_type == 'cross_validation') %>%
      group_by(instance_id) %>% 
      filter(context %in% c('doc_features_rvl', 'doc_features_bert_german')) %>% 
      filter(event_nr == max(event_nr)) %>% 
      ungroup() %>% 
      select(instance_id, context, actuals, prediction),
    by = c('instance_id', 'context')
  ) %>% 
  select(context, instance_id, actuals, prediction, event_name, type_shap, shap_value) %>% 
  write_csv('results/export/figures_f1-6.csv')
