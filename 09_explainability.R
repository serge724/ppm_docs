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

# load log data
log_df <- read_csv('log_data/process_log.csv', show_col_types = FALSE)

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

# show accuracy by instance length and event number as tile plot
type_predictions %>%
  group_by(context, fold, instance_id) %>% 
  mutate(
    correct = actuals == prediction,
    instance_length = n()
  ) %>% 
  group_by(context, instance_length, event_nr) %>% 
  summarise(pcnt_correct = sum(correct) / n(), n = n()) %>% 
  ungroup() %>% 
  mutate(event_nr = factor(event_nr), instance_length = factor(instance_length)) %>% 
  ggplot(aes(event_nr, instance_length, fill = pcnt_correct)) +
  geom_tile() +
  facet_wrap(~forcats::fct_rev(context), nrow = 2)

# load shap values for RVL and BERT-German models
shap_df <- bind_rows(
  read_csv('results/shap/doc_features_rvl/shap_df.csv', show_col_types = FALSE),
  read_csv('results/shap/doc_features_bert_german/shap_df.csv', show_col_types = FALSE)
)

# show mean absolute shap values by state name
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
  ggplot(aes(state_name, type_shap, fill = mean_abs_shap)) +
  geom_tile() +
  facet_wrap(~forcats::fct_rev(context))

# calculate and visualize shap values for individual predictions on instance level
plot_list <- list()
for (i in unique(shap_df$instance_id)) {
  predictions <- type_predictions %>% 
    filter(instance_id == i, context %in% c('doc_features_rvl', 'doc_features_bert_german')) %>% 
    filter(event_nr == max(event_nr)) %>% 
    select(context, actuals, prediction)
  
  plot_subtitle = paste0(
    'Damage type: ', unique(predictions$actuals), ' | ',
    'Prediction VGG-RVL: ', predictions[predictions$context == 'doc_features_rvl',]$prediction, ' | ',
    'Prediction BERT-German: ', predictions[predictions$context == 'doc_features_bert_german',]$prediction
  )
  
  shap_plot <- shap_df %>% 
    filter(instance_id == i) %>% 
    group_by(context, instance_id) %>% 
    mutate(event_name = paste(1:n(), state_name, sep = '. ')) %>%
    ungroup() %>% 
    select(context, event_name, file_name, c(matches('[0-9]|[0-9][0-9]'))) %>% 
    pivot_longer(c(matches('[0-9]|[0-9][0-9]')), names_to = 'type', values_to = 'shap_value') %>% 
    left_join(
      log_df %>% 
        distinct(type, type_name) %>% 
        mutate(type = as.character(type)) %>% 
        rename(type_shap = type_name),
      by = 'type'
    ) %>% 
    mutate(shap_value = ifelse(is.na(file_name), NA, shap_value)) %>% 
    ggplot(aes(event_name, type_shap, fill = shap_value)) +
    geom_tile() +
    expand_limits(x = 1:16) +
    geom_text(aes(label = round(shap_value, 2)), color = 'white', size = 2, na.rm = TRUE) +
    ggtitle(label = paste('Instance ID:', i), subtitle = plot_subtitle) +
    ylab('') +
    xlab('') +
    scale_fill_continuous(na.value = NA) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    facet_wrap(~forcats::fct_rev(context))
  plot_list %<>% append(list(shap_plot))
}
ggsave('results/shap/instance_shap_values.pdf', gridExtra::marrangeGrob(grobs = plot_list, ncol = 1, nrow = 3), width = 30, height = 30, units = 'cm')

# load calculated similarity of document embeddings
doc_similarity <- tibble()
for (i in list.files('results/embeddings/type/')) {
  doc_similarity %<>% 
    bind_rows(
      read_csv(paste0('results/embeddings/type/', i, '/doc_similarity_df.csv'), show_col_types = FALSE) %>% 
        mutate(delta = learned_similarity - pretrained_similarity, context = i)
    )
}

# show distribution of similarities by feature extractor
doc_similarity %>% 
  group_by(context) %>% 
  sample_n(50000) %>% 
  ungroup() %>% 
  pivot_longer(cols = c(pretrained_similarity, learned_similarity), 'key', 'value') %>% 
  ggplot(aes(value)) +
  geom_histogram() +
  facet_wrap(forcats::fct_rev(context) ~ forcats::fct_rev(key), nrow = 4)
