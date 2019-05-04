"""
Utilities for sqlite database data
"""
# TODO: this is an incomplete code
# written for the specific case of a 2 columns table (row_id, values)

import sqlite3


def create_db(db_name, table_name, column_names):
    """ Create sqlite database with one table and two columns. 
    :param db_name: name of the db
    :param table_name: name of the table to create
    :param column_names: list of names for the columns
    :return: 
    """
    # db creation:
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    # Create table
    c.execute("CREATE TABLE {0} ({1}, {2})".format(table_name, column_names[0], column_names[1]))

    # Save (commit) the changes and close the connection
    conn.commit()
    conn.close()


def insert_values(db_name, table_name, values):
    """ Inset a row of data in the database table 
    :param db_name: name of the tb
    :param table_name: name of the table to fill with values
    :param values: values of the row (2 values)
    :return: 
    """

    # db connection:
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    # Insert a row of data
    c.execute("INSERT INTO {0} VALUES (?,?)", (table_name, values))
    conn.commit()

    conn.close()
