from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.setup_env import setup_project_env
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

project_dir, config = setup_project_env()


class VisualiseFull:
    def __init__(self, config, df):
        self.config = config
        self.df = df
        self.fig_path = config['fig_path']

    def prepare_data_day(self, variable):
        return self.df['date'], self.df[variable]

    def prepare_data_month(self, variable):
        monthly_avg = self.df.groupby('month_year')[variable].mean()
        # idx = monthly_avg.index.to_timestamp()
        idx = pd.to_datetime(monthly_avg.index)
        return idx, monthly_avg

    def prepare_data_season(self, variable):
        season_avg = self.df.groupby('season_dt')[variable].mean()
        return season_avg.index, season_avg

    def prepare_data_year(self, variable):
        yearly_avg = self.df.groupby('year')[variable].mean()
        return yearly_avg.index, yearly_avg

    def plot_variable(self, ax, x, y, variable, time_scale, save=False):
        ax.plot(x, y, linewidth=0.5)
        ax.set_xlabel('Date' if time_scale != 'year' else 'Year')
        ax.set_ylabel(variable)
        ax.set_title(f'{variable} by {time_scale.title()}')
        ax.grid(True)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(5)

    def plot_var_full(self, time_scale, save=False):
        variables = ['avg_sea_level_pres_hpa',
                     'avg_temp_c', 'precipitation_mm']
        fig, axes = plt.subplots(nrows=len(variables),
                                 ncols=1, figsize=(18, 18))

        for ax, variable in zip(axes, variables):
            if time_scale == 'day':
                x, y = self.prepare_data_day(
                    variable)
            elif time_scale == 'month':
                x, y = self.prepare_data_month(variable)
            elif time_scale == 'season':
                x, y = self.prepare_data_season(variable)
            elif time_scale == 'year':
                x, y = self.prepare_data_year(variable)
            else:
                raise ValueError(f"Invalid times_scale: {time_scale}")

            self.plot_variable(ax, x, y, variable, time_scale)
        plt.tight_layout()
        # plt.show()
        if save:
            fig.savefig(f'{self.fig_path}_fig1')

    def plot_rolling_means(self, windows, time_scale, save=False):
        variables = ['avg_sea_level_pres_hpa',
                     'avg_temp_c', 'precipitation_mm']
        fig, axes = plt.subplots(nrows=len(variables),
                                 ncols=len(windows), figsize=(18, 12))

        for i, variable in enumerate(variables):
            for j, window in enumerate(windows):
                rolling_var = f'{variable}_rolling_{window}'
                self.df[rolling_var] = self.df[variable].rolling(
                    window=window).mean()

                if time_scale == 'year':
                    x, y = self.prepare_data_year(rolling_var)
                elif time_scale == 'season':
                    x, y = self.prepare_data_season(rolling_var)
                elif time_scale == 'month':
                    x, y = self.prepare_data_month(rolling_var)
                else:
                    raise ValueError(f"Invalid time_scale: {time_scale}")

                self.plot_variable(axes[i, j], x, y, rolling_var, time_scale)
        plt.tight_layout()
        # plt.show()
        if save:
            fig.savefig(f'{self.fig_path}_fig2')


class VisualiseByYear:
    def __init__(self, config, df):
        self.config = config
        self.df = df
        self.fig_path = config['fig_path']

    def prepare_data_month(self, df_year, variable):
        monthly_avg = df_year.groupby(df_year['date'].dt.month)[
            variable].mean()
        return monthly_avg.index, monthly_avg

    def prepare_data_season(self, df_year, variable):
        season_avg = df_year.groupby('season')[variable].mean().reindex(
            ['Winter', 'Spring', 'Summer', 'Autumn'])
        return season_avg.index, season_avg

    def prepare_data_day(self, df_year, variable):
        return df_year['day_of_year'], df_year[variable]

    def plot_variable(self, ax, x, y, variable, title, save=False):
        ax.plot(x, y, linewidth=0.5)
        ax.set_ylabel(variable)
        ax.set_title(title)
        ax.grid(True)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(5)

    def plot_axis(self, time_scale, save=False):
        variables = ['avg_sea_level_pres_hpa',
                     'avg_temp_c', 'precipitation_mm']
        fig, axes = plt.subplots(nrows=len(variables),
                                 ncols=1, figsize=(18, 12))

        for ax, variable in zip(axes, variables):
            unique_years = sorted(self.df['year'].unique())
            for year in unique_years:
                df_year = self.df[self.df['year'] == year]
                if time_scale == 'month':
                    x, y = self.prepare_data_month(df_year, variable)
                elif time_scale == 'season':
                    x, y = self.prepare_data_season(df_year, variable)
                elif time_scale == 'day':
                    x, y = self.prepare_data_day(df_year, variable)
                else:
                    raise ValueError(f"Invalid time_scale: {time_scale}")

                self.plot_variable(ax, x, y, variable, f'All {
                                   time_scale} - {time_scale.title()}')
        plt.tight_layout()
        # plt.show()
        if save:
            fig.savefig(f'{self.fig_path}_fig3')

    def plot_rolling_means(self, windows, time_scale, save=False):
        variables = ['avg_sea_level_pres_hpa',
                     'avg_temp_c', 'precipitation_mm']
        fig, axes = plt.subplots(nrows=len(variables),
                                 ncols=len(windows), figsize=(18, 12))

        for i, variable in enumerate(variables):
            for j, window in enumerate(windows):
                rolling_var = f'{variable}_rolling_{window}'
                self.df[rolling_var] = self.df[variable].rolling(
                    window=window, min_periods=1).mean()

                for year in sorted(self.df['year'].unique()):
                    df_year = self.df[self.df['year'] == year]

                    if time_scale == 'month':
                        x, y = self.prepare_data_month(df_year, rolling_var)
                    elif time_scale == 'season':
                        x, y = self.prepare_data_season(df_year, rolling_var)
                    elif time_scale == 'day':
                        x, y = self.prepare_data_day(df_year, rolling_var)
                    else:
                        raise ValueError(f"Invalid time_scale: {time_scale}")

                    self.plot_variable(axes[i][j], x, y, variable, f'All {
                                       time_scale} - Rolling {window}')
        plt.tight_layout()
        # plt.show()
        if save:
            fig.savefig(f'{self.fig_path}_fig4')


class VisualiseOther:
    def __init__(self, config, df):
        self.config = config
        self.df = df
        self.fig_path = config['fig_path']

    def plot_decade_var(self, variable, plt_type, inc_err, save=False):
        plt.figure(figsize=(18, 10))
        self.df['decade'] = (self.df['year'] // 10)*10
        decades = self.df['decade'].unique()
        if plt_type == 'line':
            for decade in decades:
                decade_data = self.df[self.df['decade'] == decade]
                monthly_stats = decade_data.groupby(
                    'month')[variable].agg(['mean', 'std'])
                if inc_err is True:
                    plt.errorbar(
                        x=monthly_stats.index, y=monthly_stats['mean'],
                        yerr=monthly_stats['std'], label=f"{decade}s", fmt='-o', linewidth=1, capsize=3, markersize=3)
                else:
                    plt.errorbar(
                        x=monthly_stats.index, y=monthly_stats['mean'],
                        label=f"{decade}s", fmt='-o', linewidth=1, capsize=3, markersize=3)
        elif plt_type == 'bar':
            bar_width = 0.08
            months = np.arange(1, 13)
            for i, decade in enumerate(decades):
                decade_data = self.df[self.df['decade'] == decade]
                monthly_stats = decade_data.groupby(
                    'month')[variable].agg(['mean', 'std'])
                if inc_err is True:
                    plt.bar(
                        months + i * bar_width, monthly_stats['mean'],
                        width=bar_width, yerr=monthly_stats['std'], label=f"{decade}s", capsize=2)
                else:
                    plt.bar(
                        months + i * bar_width, monthly_stats['mean'],
                        width=bar_width, label=f"{decade}s", capsize=2)
        else:
            raise ValueError(f"Invalid plt_type: {plt_type}")

        plt.xlabel('Month')
        plt.ylabel(f'Average {variable}')
        plt.title(f'Monthly Average {variable} by Decade (No error bar)')
        plt.xticks(range(1, 13))
        plt.legend()
        plt.grid(True)
        # plt.show()

        if save:
            plt.savefig(f'{self.fig_path}_fig5')

    def periodic_stats(self, variable, freq, stat, save=False):
        # Stores
        mean_store = {}
        std_store = {}
        cv_store = {}
        plt.figure(figsize=(18, 9))
        unique_years = self.df['year'].unique()
        for year in unique_years:
            year_data = self.df[self.df['year'] == year]
            # Means
            mean = year_data.groupby(freq)[variable].mean()
            mean_store[year] = mean
            # Stds
            std = year_data.groupby(freq)[variable].std()
            std_store[year] = std
            # CVs
            cv = (std/mean)*100
            cv_store[year] = cv

            if stat == 'mean':
                plt.plot(range(1, len(mean) + 1), mean,
                         label=str(year), linewidth=0.5)
            elif stat == 'std':
                plt.plot(range(1, len(std) + 1), std,
                         label=str(year), linewidth=0.5)
            elif stat == 'cv':
                plt.plot(range(1, len(cv) + 1), cv,
                         label=str(year), linewidth=0.5)
            else:
                raise ValueError(f'Invalid stat: {stat}')

        plt.xlabel(freq)
        plt.ylabel(f'CV of {variable}')
        plt.title(f'{freq} CV of {variable}')
        plt.grid(True)
        # plt.show()
        if save:
            plt.savefig(f'{self.fig_path}_fig6')
        return mean_store, std_store, cv_store

    def plot_grouped_years(self, variable, agg_type, group_1, group_2, save=False):
        plt.figure(figsize=(18, 12))
        # Calc means
        group_1_mean = self.df[self.df['year'].isin(group_1)].groupby('month')[
            variable].mean()
        group_2_mean = self.df[self.df['year'].isin(group_2)].groupby('month')[
            variable].mean()
        # Calc cumulative means
        group_1_cum_mean = group_1_mean.cumsum()
        group_2_cum_mean = group_2_mean.cumsum()
        if agg_type == 'mean':
            plt.plot(range(1, 13), group_1_mean, label=f'Range: {
                     group_1[0]}:{group_1[-1]}', marker='o')
            plt.plot(range(1, 13), group_2_mean, label=f'Range: {
                     group_2[0]}:{group_2[-1]}', marker='o')
        elif agg_type == 'cum_mean':
            plt.plot(range(1, 13), group_1_cum_mean, label=f'Range: {
                     group_1[0]}:{group_1[-1]}', marker='o')
            plt.plot(range(1, 13), group_2_cum_mean, label=f'Range: {
                     group_2[0]}:{group_2[-1]}', marker='o')
        else:
            raise ValueError(f'Invalid agg_type: {agg_type}')
        plt.xlabel('Month')
        plt.ylabel(f'Average {variable}')
        plt.title(f'Comparison of Monthly Mean {variable}')
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr',
                   'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.legend()
        plt.grid(True)
        # plt.show()
        if save:
            plt.savefig(f'{self.fig_path}_fig7')
