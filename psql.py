#!/usr/bin/env python

import itertools as its
import operator as op
import io
import csv

import psycopg2


def execute(config, query):
    conn = psycopg2.connect(**config['database'])
    cur = conn.cursor()
    cur.execute(query)
    desc = cur.description
    desc = map(op.itemgetter(0), desc)
    desc = tuple(desc)
    data = cur.fetchall()
    data = map(zip, its.repeat(desc), data)
    data = map(dict, data)
    data = tuple(data)
    cur.close()
    conn.close()
    return data


def read_process_table(config, names):
    # Lookup functions
    QQ = """ SET search_path TO feature;
          SELECT id, name
          FROM function
          WHERE name IN {}""".format(names)
    data = execute(config, QQ)

    ranked = enumerate(names)
    foo = []
    for ii, name in ranked:
        for datum in data:
            if name == datum['name']:
                bar = {'function_id': datum['id'],
                       'rank': ii}
                foo.append(bar)

    # Lookup process
    in_clause = map(op.methodcaller('values'), foo)
    in_clause = ', '.join('({}, {})'.format(ii, jj) for ii, jj in in_clause)
    QQ = f"""
          SET search_path TO feature;
          SELECT pc.process_id
          FROM process_composition pc
          JOIN (
          VALUES {in_clause}
          ) AS input(function_id, rank)
          ON pc.function_id = input.function_id
          AND pc.rank = input.rank
          GROUP BY pc.process_id
          HAVING COUNT(*) = (SELECT COUNT(*)
          FROM (VALUES {in_clause}) AS input(function_id, rank));
          """.format(in_clause=in_clause)

    data = execute(config, QQ)
    data = tuple(data)[0]['process_id']
    return data


def write_session(config, buffer):
    columns = ('spl_row',
               'event_id',
               'behavior_id',
               'group_id',
               'instrument_id',
               'process_id',
               'value')
    conn = psycopg2.connect(**config['database'])
    cur = conn.cursor()
    cur.execute("""SET search_path TO feature;""")
    cur.copy_from(buffer, 'values', sep=',',
                  columns=columns,
                  null='nan')
    conn.commit()
    cur.close()
    conn.close()
    return None


def load_process(config, names):
    name_str = ('__'.join(names),)
    ranked = enumerate(names)
    ranked = tuple(ranked)

    # Add functions?
    QQ = """ SET search_path TO feature;
    select id, name from function;
    """
    data = execute(config, QQ)
    name_data = map(op.itemgetter('name'), data)
    name_data = tuple(name_data)
    to_add = set(names) - set(name_data)
    to_add = tuple(to_add)
    buf = io.StringIO()
    buf.write('\n'.join(to_add))
    buf.seek(0)
    conn = psycopg2.connect(**config['database'])
    cur = conn.cursor()
    QQ = """SET search_path TO feature;"""
    cur.execute(QQ)
    cur.copy_from(buf, 'function', columns=('name',))
    buf.close()
    conn.commit()

    # Lookup function ids
    QQ = """SET search_path TO feature;
    select id, name from function;"""
    data = execute(config, QQ)
    data = tuple(data)
    foo = []
    for ii, name in ranked:
        for datum in data:
            if name == datum['name']:
                bar = {'function_id': datum['id'],
                       'rank': ii}
                foo.append(bar)

    # Lookup process
    in_clause = map(op.methodcaller('values'), foo)
    in_clause = ', '.join('({}, {})'.format(ii, jj) for ii, jj in in_clause)
    QQ = f"""
          SET search_path TO feature;
          SELECT pc.process_id
          FROM process_composition pc
          JOIN (
          VALUES {in_clause}
          ) AS input(function_id, rank)
          ON pc.function_id = input.function_id
          AND pc.rank = input.rank
          GROUP BY pc.process_id
          HAVING COUNT(*) = (SELECT COUNT(*)
          FROM (VALUES {in_clause}) AS input(function_id, rank));
          """.format(in_clause=in_clause)

    # Add process if it doesn't exist
    data = execute(config, QQ)
    pid = map(op.itemgetter('process_id'), data)
    pid = next(pid, None)
    if not pid:
        QQ = """SET search_path TO feature;
        INSERT INTO process DEFAULT VALUES
        RETURNING id;"""
        cur.execute(QQ)
        pid = cur.fetchone()[0]
        conn.commit()

    # add composition if not already exists
    QQ = """SET search_path TO feature;
    SELECT process_id
    from process_composition
    WHERE process_id = {}""".format(pid)
    data = execute(config, QQ)
    if not data:
        for each in foo:
            each['process_id'] = pid
        buf = io.StringIO()
        fieldnames = ('process_id', 'function_id', 'rank',)
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writerows(foo)
        buf.seek(0)
        cur.copy_from(buf, 'process_composition', sep=',',
                      columns=('process_id', 'function_id', 'rank'),
                      null='nan')
    conn.commit()
    buf.close()
    cur.close()
    conn.close()
    return buf

    return data

