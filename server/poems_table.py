import datetime
import random

from google.cloud import bigtable
from google.cloud.bigtable import column_family
from google.cloud.bigtable import row_filters


class Poems():
    def __init__(self):
        project_id = 'poembot'
        instance_id = 'poems101'
        self.table_id = 'poems'
        self.column_family_id = 'cf1'
        self.client = bigtable.Client(project=project_id, admin=True)
        self.instance = self.client.instance(instance_id)
        self.table = self.instance.table(self.table_id)

    def create_table(self):
        print('Creating column family cf1 with Max Version GC rule...')
        # Create a column family with GC policy : most recent N versions
        # Define the GC policy to retain only the most recent 2 versions
        max_versions_rule = column_family.MaxVersionsGCRule(2)
        column_families = {self.column_family_id: max_versions_rule}
        if not self.table.exists():
            self.table.create(column_families=column_families)
        else:
            print("Table {} already exists.".format(self.table_id))

    def delete_table(self):
        self.table.delete()

    def write_to_table(self, poems):
        rows = []
        column = 'poem'.encode()
        for i, value in enumerate(poems):
            # Note: This example uses sequential numeric IDs for simplicity,
            # but this can result in poor performance in a production
            # application.  Since rows are stored in sorted order by key,
            # sequential keys can result in poor distribution of operations
            # across nodes.
            #
            # For more information about how to design a Bigtable schema for
            # the best performance, see the documentation:
            #
            #     https://cloud.google.com/bigtable/docs/schema-design
            row_key = 'poem{}'.format(i).encode()
            row = self.table.direct_row(row_key)
            row.set_cell(self.column_family_id,
                         column,
                         value,
                         timestamp=datetime.datetime.utcnow())
            rows.append(row)
        self.table.mutate_rows(rows)

    def get_random_poem(self):
        random_n = random.randint(0, 499)
        column = 'poem'.encode()
        key = ('poem' + str(random_n)).encode()
        row_filter = row_filters.CellsColumnLimitFilter(1)
        row = self.table.read_row(key, row_filter)
        cell = row.cells[self.column_family_id][column][0]
        return cell.value.decode('utf-8')
