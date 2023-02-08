# Databricks notebook source
# MAGIC %md
# MAGIC # Context:
# MAGIC The products and operations team at Airbnb is working to grow bookings in Rio de Janeiro. They have a series of data surrounding inquiries, or potential reservations, that they would like to examine to look for ways to increase bookings on their platform. In addition to framing the problem fully, this analysis serves to answer the following questions:
# MAGIC 1) What metrics can we define to monitor the success of the team's efforts in improving the guest/host mathing process?
# MAGIC 2) Which areas (segments or subgroups) should be invested in to increase the number of successful bookings in the market?
# MAGIC 3) What other research, experiements, or approaches could help the company see beyond the data provided and understand the broader framing of matching supply and demand?

# COMMAND ----------

# MAGIC %md
# MAGIC ### Package imports

# COMMAND ----------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from datetime import timedelta 

%matplotlib inline
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Read in data

# COMMAND ----------

contacts = pd.read_csv('contacts.csv')
listings = pd.read_csv('listings.csv').drop_duplicates()
#users dataset has duplicate data, remove it.
users = pd.read_csv('users.csv', names=['id_guest_anon','country','words_profile'], header=0).drop_duplicates()
print("Users", users.shape, "Listings", listings.shape, "Contatcts", contacts.shape )

# COMMAND ----------

#check for duplicate users
users.id_guest_anon.nunique()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Merge data together, pd.merge changes order of observations by default. Not critical that things remain in the same order though.
# MAGIC There is a 1:many ratio between listings and contacts so we can use a left join to merge the two.
# MAGIC The users dataset, which had duplicates that were removed, potentially has users that are not part of the contacts list so we will read that dataset in differently.

# COMMAND ----------

contacts_listings=pd.merge(contacts, listings, on='id_listing_anon').reset_index(drop=True)
print(contacts_listings.shape)
users_check = pd.merge(contacts_listings, users,  how='outer', indicator=True, on='id_guest_anon')
print(users_check['_merge'].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC There are 8,891 users not included in the contacts dataset - after consulting with David, these additional users have no meaning to the project. However, this brings to mind an idea of how well airbnb is marketing itself in Rio because these might be lost opportunities for users. A quick google search of 'rio de janeiro accomodations' revealed that airbnb shows up on page 2. This is an immediate opportunity to investigate.

# COMMAND ----------

# MAGIC %md
# MAGIC Resumer merge keeping on instances that occur in the contacts ds.

# COMMAND ----------

ds = pd.merge(contacts_listings, users, on='id_guest_anon')
ds.info()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Derive a few variables of interest and perform other data transformations for ease in use:

# COMMAND ----------

#flag indicating whether a booking resulted in the contact (outcome var)
ds['booked'] = ds['ts_booking_at'].notnull()
#flag indicating a host approved the booking
ds['approved'] = ds['ts_accepted_at_first'].notnull()
ds['denied'] = ds['ts_accepted_at_first'].isnull()
#flag indicating host responded
ds['host_responded']= ds['ts_reply_at_first'].notnull()
ds['no_response_host']= ds['ts_reply_at_first'].isnull()
#flag indicating user walked away from an approved reservation
ds['user_walked'] = (ds['booked']==False) & (ds['approved']==True)
#convert some date variables into datetime objects for calculations

ds['checkin_dt']=pd.to_datetime(pd.to_datetime(ds['ds_checkin_first']).dt.date)
ds['checkout_dt']=pd.to_datetime(pd.to_datetime(ds['ds_checkout_first']).dt.date)
ds['inter_first_dt']=pd.to_datetime(pd.to_datetime(ds['ts_interaction_first']).dt.date)
#rename really long vars for ease
ds.rename(columns={'m_first_message_length_in_characters': 'm_first_mess'  }, inplace=True)

# COMMAND ----------

#additional RESERVATION-related variables:
#number of days in stay requested
ds['los'] = (ds['checkout_dt'] - ds['checkin_dt']) /timedelta(days=1)
#convert dates to datetime
#calculate the number of days between inquiry and checkin
ds['d2checkin'] = (ds['checkin_dt'] - ds['inter_first_dt']) /timedelta(days=1)
#calculate the number of minutes between inquiry and booking (used for understanding the booking process)
ds['m2book'] = (pd.to_datetime(ds['ts_booking_at']) - pd.to_datetime(ds['ts_interaction_first'])) /timedelta(minutes=1)
ds['new_prop'] = ds['total_reviews']==0

ds[['checkin_dt', 'checkout_dt','inter_first_dt', 'los','d2checkin','ts_booking_at','m2book', 'total_reviews','new_prop']].head()

# COMMAND ----------

print(ds['booked'].value_counts())
sns.countplot(x='booked', data=ds)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Not exactly balanced, but not terribly imbalanced.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis - examine data to see if it needs cleaning up

# COMMAND ----------

# MAGIC %md
# MAGIC First, since we have a few date variables in the dataset, lets look at the period of time the data spans to help us understand the context of the problem a litte more

# COMMAND ----------

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
locator = mdates.AutoDateLocator()
fig.suptitle("Histograms of Inquiries")
ax1.hist(ds['inter_first_dt'], bins=180)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
ax1.set(xlabel='Inquiry Date', ylabel='Frequency')

ax2.hist(ds['checkin_dt'], bins=100)
ax2.xaxis.set_major_locator(locator)
ax2.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
ax2.set(xlabel='Checkin Date', ylabel='Frequency')
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Inquiry Date (left panel) : This plot shows the dates the inquiries span with roughly one day represented by a bar. The data appears to have a restriction of a 6 month window (2016-01 to 2016-07ish). We see larger spikes towards the beginning of the year and then the frequency sort of plateaus. The data has a weekly cycle where the lowest frequencies appear to take place on the weekends (probably because people are out enjoying their lives rather than making reservations).
# MAGIC The frequencies start out high, plateau in the middle and then spike again towards the end of the time period.
# MAGIC 
# MAGIC Checkin Date (right panel) : Similarly, this plot shows frequencies of inquiries based on their intended checkin date. There are two clear spikes in February and Summer in 2016, a google search revealed relates to Carnival and the Olympics. These two events seemed to generate a giant chunk of the inquiries. The distribution here has a very long right tail, one that extends beyond the inquiry date. We may have a couple of outliers that are worth investigating.

# COMMAND ----------

#let's look at the distributions of the 'numeric' variables
cont_vars=['m_guests','m_first_mess','m_interactions','total_reviews','words_profile','los','d2checkin','m2book']
ds[cont_vars].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC Looking at this table, it appears as though we have some data issues to address:
# MAGIC - We have a missing value for number of guests, lets impute with mean. There are 0s for this as well, which isn't a valid number for guests so we will look at the frequency of these values and see if its systematic.
# MAGIC - Total reviews, days to checkin and minutes between first inquiry and booking all have negative values which we'll need to look at further to see if there is a potential bug in the calculation behind the scenes

# COMMAND ----------

#Look at the distribution of the number of guests:
sns.countplot(x='m_guests', order=range(16), data = ds)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Very few 0s and only one missing so impute with mean

# COMMAND ----------

#impute missing  and 0 values
ds['m_guests']=ds['m_guests'].fillna(ds['m_guests'].mean())

# COMMAND ----------

# MAGIC %md
# MAGIC Examine negative values that shoudln't be negative

# COMMAND ----------

#Total reviews
ds['total_reviews'].hist(bins=100)
plt.show()
ds['total_reviews'].hist(bins=100, range=[-100,100])
plt.ylim(0,100)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC This has a long right tail, and a few negative values. There shouldn't be negative number of reviews, so replace with missing and address with data team. If these values are displayed to users/guests, this can be a big turnoff in renting. 

# COMMAND ----------

reviews=ds['total_reviews']
reviews[ds['total_reviews']<0]=0
ds['total_reviews']=reviews
ds['total_reviews'].hist(bins=100)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Looks a little better. Looking at thie plot, there could potentially be some groups created from this variable. Investigating this would be a good future step, for now we'll leave the data as is.

# COMMAND ----------

# MAGIC %md
# MAGIC Number of days until requested checkin

# COMMAND ----------

#days to checkin
ds['d2checkin'].hist(bins=100)
plt.show()
#zoom in for negs
ds['d2checkin'].hist(bins=100, range=[-100,100])
plt.ylim(0,100)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Very few negs but in the first plot we can see there are others that are really far off in the future.
# MAGIC Regarding the negatives, we should drop these inquiries as it's not possible to approve/book a reservation that took place in the past.
# MAGIC We should also examine the extreme large values.

# COMMAND ----------

sorted_ds=ds.sort_values(by='d2checkin')
sorted_ds.tail()

# COMMAND ----------

# MAGIC %md
# MAGIC It looks like the last 3 entries in the dataset may have been booked with an incorrect year chosen? They are more than a year in the future, and I think by default Airbnb lets you book up to 3 years ahead,so we may want to clip these off. We can add a rule that omits requests after 2017 as it's pretty unlikely these people meant to book something 2 years away (Assumption-alert!).
# MAGIC We should also remove reservations that have erroneous dates, like those occurring in the past.

# COMMAND ----------

ds['removed']=(ds['checkin_dt'].dt.year >=2018) | (ds['checkin_dt']<ds['inter_first_dt'])
ds['removed'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC Overall we are removing 29 inquiries from the data for being out of bounds for a believable reservation (something a host would actually consider).

# COMMAND ----------

# MAGIC %md
# MAGIC Categorical data of interest

# COMMAND ----------

cat_vars=['contact_channel_first','guest_user_stage_first','room_type','listing_neighborhood','country','booked','approved','host_responded']
for x, v in enumerate(cat_vars):
    sns.countplot(ds[v])
    plt.show()

# COMMAND ----------

#Clean up the text vars to make dummy variables easier
ds['room_type'] = ds['room_type'].str.split(expand=True)[0]
ds['guest_user_stage_first'] = ds['guest_user_stage_first'].str.strip('-')

# COMMAND ----------

# MAGIC %md
# MAGIC Examine the minutes to booking variable to make sure there isn't a bug in the way the data are generated

# COMMAND ----------

ds['m2book'].hist()

# COMMAND ----------

# MAGIC %md
# MAGIC Some outliers here including negative values but they are very few. This brings up a question about whether there is overlap between the 'instant book' and 'contact me' channels. If a property has opted into 'instant book' and a user sends a message first, does the booking get tagged as an 'instant book' or 'contact me?'
# MAGIC We can look at the amount of time that's elapsed between the first inquiry and the actual booking for the various channels and see if instant book only has bookings that occur at roughly the same time as the first inquiry. If there are some that occur much after, then we can assume that thees types of situations will get logged under 'instant book.'

# COMMAND ----------

#examine the amount of time elapsing between contact for the channels to determine whether there is nested subgroups
print(ds['m2book'].describe())
grid = sns.FacetGrid(col='contact_channel_first', data=ds, sharey=False, sharex=False)
grid.map(sns.distplot, 'm2book')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC There isn't always an immediate booking taking place for the 'instant book' listings. Since 'instant book' properties can also have the 'contact me' channel, there is a concern that if a guest initiates an inquiry as 'contact me' for an 'instant book' property that the entry will show up as 'contact me.' Looking at the data, this doesn't seem to be the case and properties that subscribe to this feature show up as 'instant book' in the dataset regardless of whether the guest chose to contact the host first.

# COMMAND ----------

#Create a dataset with these few records omitted
ads=ds.loc[ds.removed==False]
ads.to_csv('working_ds.csv')
ads.shape

# COMMAND ----------

# MAGIC %md
# MAGIC Regenerate the plot with the data cleaned

# COMMAND ----------

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
locator = mdates.AutoDateLocator()
fig.suptitle("Histograms of Inquiries")
ax1.hist(ads['inter_first_dt'], bins=180)
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
ax1.set(xlabel='Inquiry Date', ylabel='Frequency')

ax2.hist(ads['checkin_dt'], bins=100)
ax2.xaxis.set_major_locator(locator)
ax2.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
ax2.set(xlabel='Checkin Date', ylabel='Frequency')
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC This is our final data

# COMMAND ----------

# MAGIC %md
# MAGIC # Explore the data to identify segments/relationships
# MAGIC First, let's define the problem fully and see if we can identify potential areas for improvement.
# MAGIC Then, identify segments where the matching is successful and ones where it isn't.
# MAGIC 
# MAGIC Functions to automate repeated code:

# COMMAND ----------

#This function will provide a bar plot of the grouped data
def bookingplot(dsn, varn, titlen, width, height):
    fig, ax = plt.subplots(figsize=(width,height))
    fig.suptitle(f"Inquries by {titlen}")
    #lower portion of bars showing those that were booked successfully
    ax.bar(dsn[varn], dsn['n_bookings'], label='Successful' )
    ax.bar(dsn[varn], dsn['lost_opps'], bottom= dsn['n_bookings'], label='Unsuccessful')
    ax.tick_params('x', labelrotation=45)
    ax.set(xlabel=titlen, ylabel='Frequency')
    ax.legend()
    plt.show()

# COMMAND ----------

#This function aggregates the data together according to a specific column or columns. It returns tall and wide versions of the data.
def createbookgroups(varn):
    #look at the total number of bookings by a specified category or categories
    booking_by_var=ads.groupby(varn)['booked'].agg(sum).to_frame('freq').reset_index()
    booking_by_var['type']='bookings'
    inq_by_var=pd.DataFrame(ads.groupby(varn).size().to_frame('freq').reset_index())
    inq_by_var['type'] = 'inquiries'
    b_var_stack=pd.DataFrame(pd.concat([booking_by_var, inq_by_var], axis=0))
    b_var_short=pd.DataFrame(booking_by_var.rename(columns={'freq': 'n_bookings'}).merge(inq_by_var.rename(columns = {'freq': 'n_inquiries'}), on=varn, how='outer'))
    b_var_short['book_pct'] = round(b_var_short['n_bookings'] / b_var_short['n_inquiries'],3)
    b_var_short['lost_opps'] = b_var_short['n_inquiries'] - b_var_short['n_bookings']
    b_var_short.drop(['type_x','type_y'], axis=1, inplace=True)
    return b_var_stack, b_var_short

# COMMAND ----------

inter_dt_stack, inter_dt_short = createbookgroups('inter_first_dt')
bookingplot(inter_dt_short, 'inter_first_dt', 'Date', 20, 10)

# COMMAND ----------

# MAGIC %md
# MAGIC One metric we can use is the number of bookings per day (the blue bars in the plot above). Specifically, this would be the number of successful bookings taking place in a calendar day according to local time (00:00 - 23:59:59).

# COMMAND ----------

# MAGIC %md
# MAGIC Perform some modeling to identify feature importance.

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
import statsmodels.formula.api as smf

# COMMAND ----------

#create dummy vars for categorical data
cat_vars = ['room_type', 'guest_user_stage_first', 'contact_channel_first']
labs = ['room','guest','contact']
for i, var in enumerate(cat_vars):
    ads[var].replace(' ', '_')
    ads[var].replace('-', '')    
    cat_list = 'var' + '_' + var
    cat_list = pd.get_dummies(ads[var], prefix=labs[i])
    ads1=ads.join(cat_list)
    ads=ads1

# COMMAND ----------

pred_features = ['contact_contact_me','contact_book_it', 'room_Entire', 'room_Private', 'guest_new', 'guest_past_booker', 'words_profile', 'total_reviews', 'm_first_mess', 'los', 'm_guests']
X=ads.loc[:,pred_features]
Y=ads['booked']
X.describe()

# COMMAND ----------

#Create train test splits
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=1999)
X_train.shape

# COMMAND ----------

# MAGIC %md
# MAGIC Employ some machine and statistical learning methods to identify variables that are important to the booking status.
# MAGIC First, try a Lasso (for classification) model as it is often helpful with feature removal. Since it is based on logistic (linear) regression, the assumption is that there is a linear or logistic relationship betweeen the predictors and outcome variable, which isn't an assumption that's been fully tested. As a result, we won't take these results too seriously, but will instead use them to confirm the results of our second approach, Random Forest.
# MAGIC 
# MAGIC We'll use just the training data in case time permits and we want to look at the performance of the model later.

# COMMAND ----------

lr_model = LogisticRegression(penalty='l1', solver='liblinear')
lr_model.fit(X_train, Y_train)
lr_importance = lr_model.coef_[0]
lr_importance
for i,v in enumerate(lr_importance):
    print('Feature: ', pred_features[i], ' Score: %.5f' % (v))
# plot feature importance
plt.bar([x for x in range(len(lr_importance))], lr_importance,  tick_label=pred_features)
plt.xticks(rotation=45, horizontalalignment='right')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The contact channels and status of the user/guest seem to have the most impact here. Inquiries originating through the 'contact me' channel are more negatively associated with being 'booked' compared to the control (instant book) and book it channel. This first finding isn't too surprising since all instant bookings end up being booked. What is interesting is the difference between the book it and contact me channels. This may be a first segment that we could explore.
# MAGIC We also want to look at the new and past booker variables in greater detail, both of which are more postively associated with being booked compared to the 'unknown' category.

# COMMAND ----------

# MAGIC %md
# MAGIC Use Random Forest Classifier

# COMMAND ----------

rf_model = RandomForestClassifier()
# fit the model
rf_model.fit(X_train, Y_train)
# get importance
rf_importance = rf_model.feature_importances_

# summarize feature importance
for i,v in enumerate(rf_importance):
    print('Feature: ', pred_features[i], ' Score: %.5f' % (v))
# plot feature importance
plt.bar([x for x in range(len(rf_importance))], rf_importance, tick_label=pred_features)
plt.xticks(rotation=45, horizontalalignment='right')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The results of the Random Forest model show that the contact channel is very important as well as the total number of reviews and length of the first message. A little bit of a discrepency compared to the logstic regression but that is more than likely due to the assumption of linearity. A nonparametric approach has revealed better insights into the data in this case.

# COMMAND ----------

# MAGIC %md
# MAGIC Let's explore the important features identified by the algorithms:

# COMMAND ----------

stack, short = createbookgroups('contact_channel_first')
bookingplot(short, 'contact_channel_first', 'Contact Channel',  10,5)

# COMMAND ----------

# MAGIC %md
# MAGIC Clearly concat me and book it aren't as successful as the instant book which isn't surprising
# MAGIC Flow for 'contact me' and book it channels: first contact -> host responds to request  -> host approves/denies -> for contact me, user can book or not 
# MAGIC Lets review how this is reflected in the data.

# COMMAND ----------

pd.crosstab(ads.contact_channel_first, ads.booked)

# COMMAND ----------

#Create a new outcome variable that inclues all of the possible ways a booking can succeed or fail
inq_outcome=pd.Series('Booked', index=ads.index)
inq_outcome[ads.denied==True]='Denied'
inq_outcome[(ads.approved==True) & (ads.booked==False)]='Guest Changed Mind'
inq_outcome[ads.no_response_host==True]='No Response from Host'
ads['inq_outcome']=inq_outcome
ads.inq_outcome.value_counts()

# COMMAND ----------

#establish order of axis
out_order=['No Response from Host','Denied','Guest Changed Mind','Booked']
fig, ax = plt.subplots(figsize=(10,7))
ax=sns.countplot(x='inq_outcome', data=ads, hue='contact_channel_first', order=out_order)
fig.suptitle("Inquiry Outcomes")
ax.tick_params('x', labelrotation=45)
ax.set(xlabel='Outcome', ylabel='Frequency')
ax.legend(fontsize = 12, \
               bbox_to_anchor= (1.01, 1), \
               title="Contact Channel", \
               title_fontsize = 15)
for rect in ax.patches:
    ax.text (rect.get_x() + rect.get_width()  /
             2,rect.get_height()+ 0.25,rect.get_height(),horizontalalignment='center', fontsize = 9)
plt.show()

# COMMAND ----------

# of denials
(3262+6170) /27858

# COMMAND ----------

# MAGIC %md
# MAGIC Let's look at denials to see if the user status, number of reviews, and first message length make a difference (from RF and Lasso)

# COMMAND ----------

#new_user status

approval=pd.Series('Other', index=ads.index)
approval[(ads.approved==True)]='Approved'
approval[(ads.denied==True)]='Denied'
approval[ads.contact_channel_first=='instant_book']='Instant Book'
approval[ads.no_response_host==True]='Other'

ads['approval']=approval
ads.approval.value_counts()

out_order=['Instant Book','Approved','Denied']
fig, ax = plt.subplots(figsize=(10,7))
ax=sns.countplot(x='approval', data=ads, hue='guest_user_stage_first', order=out_order)
fig.suptitle("Inquiry Outcomes")
ax.tick_params('x', labelrotation=45)
ax.set(xlabel='Outcome', ylabel='Frequency')
ax.legend(fontsize = 12, \
               bbox_to_anchor= (1.01, 1), \
               title="User Status", \
               title_fontsize = 15)
for rect in ax.patches:
    ax.text (rect.get_x() + rect.get_width()  /
             2,rect.get_height()+ 0.25,rect.get_height(),horizontalalignment='center', fontsize = 9)
plt.show()

# COMMAND ----------

out_order=['Instant Book','Approved','Denied']
fig, ax = plt.subplots(figsize=(10,7))
ax=sns.boxplot(y='approval', x='total_reviews', data=ads, hue='contact_channel_first', order=out_order, orient='h', showfliers=False)
fig.suptitle("Inquiry Outcomes")
ax.tick_params('x', labelrotation=45)
ax.set(xlabel='Outcome', ylabel='Total Reviews')
ax.legend(fontsize = 12, \
               bbox_to_anchor= (1.01, 1), \
               title="Contact Channel", \
               title_fontsize = 15)
for rect in ax.patches:
    ax.text (rect.get_x() + rect.get_width()  /
             2,rect.get_height()+ 0.25,rect.get_height(),horizontalalignment='center', fontsize = 9)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 'New' properties, which we can infer as having 0 total reviews, tend to be using the Contact Me channel. Instant book users have many more total reviews, but that's not surprising since all of their inquries are approved.
# MAGIC I think this variable is related, but not in a way that we could manipulate to move the needle. There is very weak evidence here that shows that new hosts aren't comfortable with the 'instant book' feature. Also, new users might be more inclined to deny. This is an opportunity for improvement.

# COMMAND ----------

out_order=['Instant Book','Approved','Denied']
fig, ax = plt.subplots(figsize=(10,7))
ax=sns.boxplot(y='approval', x='m_first_mess', data=ads, hue='contact_channel_first', order=out_order, orient='h', showfliers=False)
fig.suptitle("Inquiry Outcomes")
ax.tick_params('x', labelrotation=45)
ax.set(xlabel='Outcome', ylabel='Total Reviews')
ax.legend(fontsize = 12, \
               bbox_to_anchor= (1.01, 1), \
               title="Contact Channel", \
               title_fontsize = 15)
for rect in ax.patches:
    ax.text (rect.get_x() + rect.get_width()  /
             2,rect.get_height()+ 0.25,rect.get_height(),horizontalalignment='center', fontsize = 9)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC No real discernable difference here. This should probably be looked into as to why this is important as  feature...

# COMMAND ----------

# MAGIC %md
# MAGIC Other features of interest

# COMMAND ----------

room_stack, room_short = createbookgroups('room_type')
bookingplot(room_short, 'room_type', 'Room Type',  10,5)

# COMMAND ----------

hood_stack, hood_short = createbookgroups('listing_neighborhood')
bookingplot(hood_short, 'listing_neighborhood', 'Neighborhood',  30,7)

# COMMAND ----------

# MAGIC %md
# MAGIC By Country

# COMMAND ----------

stack, short = createbookgroups('country')
bookingplot(short, 'country', 'country',  30,5)

# COMMAND ----------

stack, short = createbookgroups('guest_user_stage_first')
print(ads['guest_user_stage_first'].value_counts())
#subset out the 'unknowns'
short = short.loc[short['guest_user_stage_first']!='-unknown-']
bookingplot(short, 'guest_user_stage_first', 'guest_user_stage_first',  10,5)

# COMMAND ----------

sns.set_theme(style="whitegrid")
ax=sns.boxplot(x='booked',y='d2checkin', data = ads)

# COMMAND ----------

# MAGIC %md
# MAGIC Is there a significant difference between the two?

# COMMAND ----------

sns.set_theme(style="whitegrid")
ax=sns.boxplot(x='booked',y='total_reviews', data = ads)

# COMMAND ----------

sns.set_theme(style="whitegrid")
ax=sns.boxplot(x='booked',y='los', data = ads)
