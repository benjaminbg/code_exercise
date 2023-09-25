#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 12:41:05 2023

@author: mlops
"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer

class DataFormatterTransformer(BaseEstimator, TransformerMixin):
    def _init_(self):
        self.used_likely_word_list = ['open box', 'semi nuevo', 'poco uso', 'funciona',
                                      'perfecto', 'impecable', 'optimo', 'garantia', 'mes uso',
                                      'meses uso'	,'excelente', 'leer descripcion']
        pass

    def process_payment_methods(self,series):
      item_list = []
      for items in series:
          item_list.append([value for key, value in items.items() if key == "description"])
      return set([item for sublist in item_list for item in sublist])

    def extract_payment_methods(self,df):
      df_copy = df.copy()
      df_copy["non_mercado_pago_payment_methods_exploded"] = df_copy["non_mercado_pago_payment_methods"].apply(lambda x: self.process_payment_methods(x))
      df_copy['pm_efectivo'] = df_copy["non_mercado_pago_payment_methods_exploded"].apply(lambda x: 'Efectivo' in x)
      tarjetas_credito = ['Tarjeta de crédito', 'Diners', 'Visa', 'American Express','Visa Electron', 'Mastercard Maestro','MasterCard']
      df_copy['pm_tarjeta'] = df_copy["non_mercado_pago_payment_methods_exploded"].apply(lambda x: any(item in x for item in tarjetas_credito))
      df_copy['pm_transferencia'] = df_copy["non_mercado_pago_payment_methods_exploded"].apply(lambda x: 'Transferencia bancaria' in x)
      df_copy['pm_mercadoPago'] = df_copy["non_mercado_pago_payment_methods_exploded"].apply(lambda x: 'MercadoPago' in x)
      df_copy['pm_acordar_con_el_comprador'] = df_copy["non_mercado_pago_payment_methods_exploded"].apply(lambda x: 'Acordar con el comprador' in x)
      del df_copy["non_mercado_pago_payment_methods_exploded"]
      return df_copy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy = self.extract_payment_methods(X_copy)
        X_copy["title_contains_likely_used_word"] = X_copy["title"].apply(
            lambda x: any(word in x.lower() for word in self.used_likely_word_list)
            )

        columns_to_delete = ['date_created','last_updated','original_price','category_id','international_delivery_mode','shipping.free_methods',
                             'differential_pricing','official_store_id','deal_ids','permalink','subtitle' ,
                             'catalog_product_id','video_id','secure_thumbnail','title','thumbnail','id',
                             'shipping.methods','seller_address.city.name','seller_address.state.name',
                             'seller_address.country.name','pictures', 'descriptions','listing_source',
                             'coverage_areas','warranty','sub_status','non_mercado_pago_payment_methods',
                             'seller_id','variations','shipping.dimensions','attributes','tags','parent_item_id',
                             'shipping.tags']

        for column in columns_to_delete:
          del X_copy[column]

        return X_copy
    
    def test():
        return 'test_transformer'


class LabelEncoderTransformer(BaseEstimator, TransformerMixin):

    def _init_(self):
        self.label_encoders = {}
        self.categorical_columns = ['status','currency_id','seller_address.city.id','seller_address.state.id','seller_address.country.id','site_id','listing_type_id','shipping.mode', 'buying_mode', 'shipping.free_shipping']

    def fit(self, X, y=None):
        label_encoded_X = X.copy()
        for column in self.categorical_columns:
            le = LabelEncoder()
            label_encoded_X[column] = le.fit(X[column])
            self.label_encoders[column] = le
        return self

    def transform(self, X, y=None):
        label_encoded_X = X.copy()
        for column in self.categorical_columns:
            le = self.label_encoders[column]
            label_encoded_X[column] = le.transform(X[column])
        return label_encoded_X

