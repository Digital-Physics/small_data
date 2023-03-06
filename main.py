import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_graphs(list_of_lists):
    """this function will stack 10 bar charts and save it to a png"""
    fig, axs = plt.subplots(10, 1, figsize=(8, 32), sharex=False)

    for i, element in enumerate(list_of_lists):
        # unpack each element. x are the feature descriptions. y are the coefficient values of the fitted Lasso model.
        company, target, x, y, r_squared, drop = element
        # we split positive and negatives out to color these coefficients differently in the bar graph
        pos_x_axis_idx = np.where(y >= 0)[0]
        neg_x_axis_idx = np.where(y < 0)[0]

        axs[i].bar(pos_x_axis_idx, y[y >= 0])
        axs[i].bar(neg_x_axis_idx, y[y < 0], color="red")
        axs[i].set_title(f"{company}_{target}_drop_{drop}_R^2_{r_squared}")
        axs[i].set_ylim([min(y)-1, max(y)+1])
        axs[i].set_xticks(range(len(x)))
        axs[i].set_xticklabels(x, rotation=90)

    plt.subplots_adjust(hspace=4.5)
    plt.savefig(f"Key_drivers_of_{target}_feature_drop_{drop}.png")
    plt.clf()


def grid_search(grid_alphas, drop_multicollinearity_fields, target_row_idx):
    """we do some 1-d grid search over the coefficient-penalization parameter in the Lasso Regression model.
    The higher the alpha, the more parsimonious the trained model will end up being...
    This will help us zero-in on (literally. the fitted model's feature coefficients will go to zero) key drivers"""
    df = pd.read_excel("./noise_and_signal.xlsx", "input_tab")

    companies = ['apple', 'microsoft', 'amazon', 'twitter', 'openai', 'facebook', 'alphabet', 'netflix', 'tesla', 'philadelphia philms']
    company_cols = {}
    for company in companies:
        company_cols[company] = [col for col in df.columns if company in col]

    feature_rows = [i for i in range(df.shape[0] - 3)]
    features_plus_target_rows = feature_rows + [target_row_idx]

    # initialize
    best_total_r_squared = float("-inf")
    best_alpha_penalty = None
    best_fit_details = {"target": [df["field name"].loc[target_row_idx]]}
    good_individual_r_squared = []

    for alpha in grid_alphas:
        total_r_squared = 0
        coms = []
        y_hats = []
        y_tests = []
        test_pred_residuals = []
        y_hat_inverse_scales = []
        y_test_inverse_scales = []
        intercepts = []
        coefficients = []
        descriptions = []
        dropped_desc = []
        leftover_descriptions = []
        r_squareds = []
        percent_within_20_percent = []

        for company in companies:
            # transpose our features from rows to columns
            com_data_x = df[company_cols[company]].iloc[feature_rows].transpose()
            com_data_y = df[company_cols[company]].iloc[target_row_idx].transpose()
            field_names = df["field name"].iloc[features_plus_target_rows].transpose()

            # split our data early to avoid data leakage
            com_x_train, com_x_test, com_y_train, com_y_test = train_test_split(com_data_x, com_data_y, test_size=0.2)

            # convert train and test to dataframe for correlation method
            com_x_train_df = pd.DataFrame(com_x_train)
            com_x_test_df = pd.DataFrame(com_x_test)

            # initialize in case it is not created
            to_drop = []

            if drop_multicollinearity_fields:
                corr_matrix = com_x_train_df.corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                # 0.6 is not a very high bar. hopefully some of the noise will have this much correlation so we can see the dropping work.
                to_drop = [col for col in upper.columns if any(upper[col] > 0.6)]
                com_x_train_df = com_x_train_df.drop(com_x_train_df[to_drop], axis=1)
                com_x_test_df = com_x_test_df.drop(com_x_test_df[to_drop], axis=1)

            # normalize the data
            # scaler objects get mean and standard deviation and have helpful methods like the inverse which we'll use later
            scalerx = StandardScaler().fit(com_x_train_df)
            scalery = StandardScaler().fit(com_y_train.to_frame(name="target"))
            # use the training data metrics to scale training and test data
            # convert the data series input into a dataframe and squeeze out the dimension of 1
            com_x_train_df = scalerx.transform(com_x_train_df)
            com_y_train = scalery.transform(com_y_train.to_frame(name="target")).squeeze()
            com_x_test_df = scalerx.transform(com_x_test_df)
            com_y_test = scalery.transform(com_y_test.to_frame(name="target")).squeeze()

            # create Lasso model object then feed it the data and fit the model
            lasso = Lasso(alpha=alpha, max_iter=5000)
            lasso.fit(com_x_train_df, com_y_train)

            # make predictions with fitted model on our test data
            com_y_predict = lasso.predict(com_x_test_df)
            # transform the prediction back to numbers we understand
            com_y_predict_inverse_scale = scalery.inverse_transform(com_y_predict.reshape(-1, 1))
            # com_y_predict_inverse_scale = scalery.inverse_transform(com_y_predict)
            # put the test target in a more intuitive form as well
            com_y_test_inverse_scale = scalery.inverse_transform(com_y_test.reshape(-1, 1))
            # com_y_test_inverse_scale = scalery.inverse_transform(com_y_test)

            # R**2
            residuals = com_y_predict - com_y_test
            ss_residuals = np.sum(residuals**2)
            ss_total = sum([(val - sum(com_y_test)/len(com_y_test))**2 for val in com_y_test])
            r_squared = 1 - ss_residuals/(ss_total + 0.0000000001)
            total_r_squared += r_squared

            # log
            # we look for the best alpha that works for all companies in terms of total R**2 metric...
            # this acts like a regularization technique in some sense...
            # but we also write the the good, lower level, company-alpha run details to a text file...
            # even if that alpha didn't work for all companies
            if 0.9 < r_squared < 1:
                drop_desc_ = [field_names[i] for i in to_drop]
                leftover_desc_ = [n for n in field_names if n not in drop_desc_ and n != df["field name"].iloc[target_row_idx]]
                good_coef_idx_ = [i for i, coef in enumerate(lasso.coef_) if abs(coef) > 0.1]
                good_coef_desc_ = [leftover_desc_[i] for i in good_coef_idx_]
                good_individual_r_squared.append((r_squared,
                                                 df["field name"].iloc[target_row_idx], 
                                                 company,
                                                 alpha,
                                                 drop_multicollinearity_fields,
                                                 good_coef_desc_))
            
            # custom metric
            # test set predictions within 20% of the true amount
            percent_diff = np.abs(residuals/(com_y_test + 0.000000000001))
            custom = (percent_diff < 0.2).sum() / len(com_y_test)
            
            # append details into lists for the specific company
            coms.append(company)
            y_hats.append(com_y_predict)
            y_tests.append(com_y_test)
            test_pred_residuals.append(residuals)
            y_hat_inverse_scales.append(com_y_predict_inverse_scale)
            y_test_inverse_scales.append(com_y_test_inverse_scale)
            intercepts.append(lasso.intercept_)
            coefficients.append(lasso.coef_)
            descriptions.append(field_names)
            drop_desc = [field_names[i] for i in to_drop]
            dropped_desc.append(drop_desc)
            leftover_descriptions.append([n for n in field_names if n not in drop_desc and n != df["field name"].iloc[target_row_idx]])
            r_squareds.append(r_squared)
            percent_within_20_percent.append(custom)

        # this R**2 total is the sum of four individual R**2 for a given alpha
        if total_r_squared > best_total_r_squared:
            best_total_r_squared = total_r_squared
            best_alpha_penalty = alpha
            best_fit_details = {
                "coms": coms,
                "target": [df["field name"].iloc[target_row_idx]]*len(coms),
                "y_hats": y_hats,
                "y_tests": y_tests,
                "test_pred_residuals": test_pred_residuals,
                "y_hat_inverse_scales": y_hat_inverse_scales,
                "y_test_inverse_scales": y_test_inverse_scales,
                "intercepts": intercepts,
                "coefficients": coefficients,
                "descriptions": descriptions,
                "dropped_desc": dropped_desc,
                "leftover_descriptions": leftover_descriptions,
                "r_squareds": r_squareds,
                "percent_within_20_percent": percent_within_20_percent}

    # prepare the details of the best alpha results in a format for printing out
    list_of_lists = []
    for key, value in best_fit_details.items():
        list_of_lists.append([(key, val) for val in value])
    zipped_dict_values = zip(*list_of_lists)

    # write report to txt file and create visualization in pngs
    # the pngs will allow you to quickly and visualize important features for the model across all companies
    # the txt report has additional details
    with open(f"report_{best_fit_details['target'][0]}.txt", "a+") as f:
        input_list_to_graph = []
        f.write('\n')
        f.write(f"dropped multicollinearity fields: {drop_multicollinearity_fields}")
        f.write('\n')
        f.write(f"best alpha coeff penalty weight, after using our test set like a cross val set: {best_alpha_penalty}")
        f.write('\n')
        f.write('best overall, across all companies, hyper-parameter alpha results for the specified target:')
        f.write('\n')
        for tup in zipped_dict_values:
            f.write('\n')
            # com, target, x, y, R**2, drop = inputs_to_graph
            # x are the (leftover after any dropped fields) feature descriptions for the bar graph
            # y are the coefficient values of the fitted model for the bar graph
            inputs_to_graph = [None] * 5
            for item in tup:
                if item[0] == "coms":
                    inputs_to_graph[0] = item[1]
                elif item[0] == "target":
                    inputs_to_graph[1] = item[1]
                elif item[0] == "leftover_descriptions":
                    inputs_to_graph[2] = item[1]
                elif item[0] == "coefficients":
                    inputs_to_graph[3] = item[1]
                elif item[0] == "r_squareds":
                    inputs_to_graph[4] = item[1]
                f.write(str(item))
                f.write('\n')

            input_list_to_graph.append(inputs_to_graph + [drop_multicollinearity_fields])

        generate_graphs(input_list_to_graph)

    # write best R**2s to one file
    with open(f"best_individual_company-alpha_results.txt", "a+") as f:
        for el in good_individual_r_squared:
            f.write("R**2, target, company, alpha, drop, significant coefficient descriptions:")
            f.write('\n')
            f.write(f"{el}")
            f.write('\n')


def run():
    """loop over the model variations and kick off each run."""
    for drop_fields in [False, True]:
        for target_idx in range(18, 21):
            print("run details: drop fields?, target idx:", drop_fields, target_idx)
            grid_search([0.001, 0.01, 0.03, 0.7, 0.15, 0.3, 0.5, 1], drop_fields, target_idx)


if __name__ == "__main__":
    run()




