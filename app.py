import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve

st.set_page_config(page_title='HR Analytics AI',page_icon='📊',layout='wide')
ROOT=Path(__file__).parent
R=ROOT/'resultados'
st.title('HR Analytics AI')
st.caption('Ciencia de Datos | Rotación del talento humano ')
page=st.sidebar.radio('Secciones',['Resumen ejecutivo','Análisis exploratorio','Análisis estadístico','Inteligencia artificial','Simulador predictivo'])
name='WA_Fn-UseC_-HR-Employee-Attrition.csv'
upload=st.sidebar.file_uploader('Cargar CSV original',type='csv')
df=None
if upload is not None or (ROOT/'data'/name).exists():
    try:
        df=pd.read_csv(upload if upload is not None else ROOT/'data'/name)
        required={'Attrition','Department','JobRole','MonthlyIncome','Age','YearsAtCompany','OverTime','JobSatisfaction','TotalWorkingYears','DistanceFromHome','YearsInCurrentRole'}
        if not required.issubset(df.columns) or set(df.Attrition.dropna().unique())!={'Yes','No'}: raise ValueError('El CSV no corresponde al esquema IBM esperado.')
        df=df.drop_duplicates().dropna(subset=['Attrition']).copy(); df['Salida']=df.Attrition.map({'Yes':1,'No':0})
    except Exception as e:
        st.error(str(e)); df=None
if page=='Resumen ejecutivo':
    n=len(df) if df is not None else 1470; exits=int(df.Salida.sum()) if df is not None else 237
    for col,label,value in zip(st.columns(4),['Empleados','Salidas','Permanencia','Proporción de salida'],[n,exits,n-exits,f'{exits/n:.2%}']): col.metric(label,value)
    st.plotly_chart(px.pie(names=['Salida','Permanencia'],values=[exits,n-exits],hole=.6,color_discrete_sequence=['#e69f38','#178d91']),use_container_width=True)
    st.info('Sin CSV se muestran los indicadores originales documentados. Las métricas e hipótesis exportadas corresponden al estudio original.')
elif page=='Análisis exploratorio':
    if df is None: st.warning('Carga el CSV original para activar los filtros.'); st.stop()
    dept=st.multiselect('Departamento',sorted(df.Department.unique()),default=sorted(df.Department.unique()))
    roles=st.multiselect('Cargo',sorted(df.JobRole.unique()),default=sorted(df.JobRole.unique()))
    ot=st.multiselect('Horas extras',sorted(df.OverTime.unique()),default=sorted(df.OverTime.unique()))
    f=df[df.Department.isin(dept)&df.JobRole.isin(roles)&df.OverTime.isin(ot)]
    if f.empty: st.warning('No hay registros para estos filtros.'); st.stop()
    st.metric('Proporción de salida filtrada',f'{f.Salida.mean():.2%}',f'{len(f)} registros',delta_color='off')
    a=f.groupby('JobRole',as_index=False).Salida.mean(); a.Salida*=100
    st.plotly_chart(px.bar(a,x='JobRole',y='Salida',labels={'Salida':'Salida (%)'}),use_container_width=True)
    st.plotly_chart(px.box(f,x='Attrition',y='MonthlyIncome',color='Attrition'),use_container_width=True)
    st.plotly_chart(px.scatter(f,x='Age',y='MonthlyIncome',color='Attrition',facet_col='Department'),use_container_width=True)
    st.download_button('Descargar datos filtrados',f.to_csv(index=False),'datos_filtrados.csv')
elif page=='Análisis estadístico':
    st.subheader('Pruebas de hipótesis — estudio original'); st.dataframe(pd.read_csv(R/'hipotesis.csv'),hide_index=True)
    if df is not None:
        cols=['Age','MonthlyIncome','YearsAtCompany','TotalWorkingYears','DistanceFromHome','YearsInCurrentRole']
        st.plotly_chart(px.imshow(df[cols+['Salida']].corr(),color_continuous_scale='RdBu_r',zmin=-1,zmax=1),use_container_width=True)
        z=StandardScaler().fit_transform(df[cols].fillna(df[cols].median())); p=PCA(n_components=2); xy=p.fit_transform(z)
        st.metric('Varianza explicada PCA',f'{p.explained_variance_ratio_.sum():.2%}')
        st.plotly_chart(px.scatter(x=xy[:,0],y=xy[:,1],color=df.Attrition,labels={'x':'PC1','y':'PC2'}),use_container_width=True)
elif page=='Inteligencia artificial':
    m=pd.read_csv(R/'metricas.csv'); st.dataframe(m,hide_index=True)
    st.plotly_chart(px.bar(m.melt(id_vars='Modelo'),x='variable',y='value',color='Modelo',barmode='group'),use_container_width=True)
    st.subheader('Validación cruzada'); st.dataframe(pd.read_csv(R/'validacion.csv'),hide_index=True)
    imp=pd.read_csv(R/'importancia.csv',index_col=0).head(12).sort_values('Importancia'); st.plotly_chart(px.bar(imp,x='Importancia',y=imp.index,orientation='h',title='Importancia por permutación (no causal)'),use_container_width=True)
    pred=pd.read_csv(R/'predicciones.csv'); cm=confusion_matrix(pred.Real,pred.Prediccion,labels=[0,1]); st.plotly_chart(px.imshow(cm,text_auto=True,x=['No','Sí'],y=['No','Sí'],labels={'x':'Predicción','y':'Real'}),use_container_width=True)
    fpr,tpr,_=roc_curve(pred.Real,pred.Puntuacion); st.plotly_chart(px.line(x=fpr,y=tpr,labels={'x':'Falsos positivos','y':'Verdaderos positivos'},title='ROC — modelo seleccionado'),use_container_width=True)
else:
    st.warning('Uso académico: puntuación no calibrada; no utilizar para decisiones individuales de empleo.')
    if df is None: st.info('Carga el CSV original para entrenar la regresión logística y activar el simulador.'); st.stop()
    @st.cache_resource
    def train(data):
        X=data.drop(columns=['Attrition','Salida','Age','Gender','MaritalStatus','EmployeeNumber','EmployeeCount','Over18','StandardHours'],errors='ignore'); y=data.Salida
        X=X.loc[:,X.nunique()>1]; tr,_,yt,_=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
        nc=tr.select_dtypes(include='number').columns.tolist(); cc=tr.select_dtypes(exclude='number').columns.tolist()
        prep=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nc),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cc)])
        model=Pipeline([('prep',prep),('model',LogisticRegression(max_iter=2000,class_weight='balanced'))]).fit(tr,yt)
        base={c:tr[c].median() if c in nc else tr[c].mode().iloc[0] for c in tr.columns}; return model,tr,base
    model,tr,base=train(df); row=base.copy()
    for c in ['MonthlyIncome','YearsAtCompany','JobSatisfaction']:
        row[c]=st.slider(c,int(tr[c].min()),int(tr[c].max()),int(base[c]))
    row['OverTime']=st.selectbox('Horas extras',sorted(tr.OverTime.unique()))
    row['Department']=st.selectbox('Departamento',sorted(tr.Department.unique()))
    row['JobRole']=st.selectbox('Cargo',sorted(tr.loc[tr.Department==row['Department'],'JobRole'].unique()))
    score=model.predict_proba(pd.DataFrame([row],columns=tr.columns))[0,1]; st.metric('Puntuación estimada de salida',f'{score:.1%}'); st.progress(float(score)); st.caption('Modelo reentrenado con el CSV cargado; los demás atributos se fijan en valores típicos del entrenamiento. Puede diferir de Colab.')
