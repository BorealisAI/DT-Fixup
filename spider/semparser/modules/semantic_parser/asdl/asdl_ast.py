# Copyright (c) 2020-present, Royal Bank of Canada.
# Copyright (c) 2017-present, Pengcheng Yin
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#################################################################################################
# Code is based on the tranX (https://arxiv.org/abs/1810.02720) implementation
# from https://github.com/pcyin/tranX by Pengcheng Yin
#################################################################################################

from io import StringIO
from semparser.modules.semantic_parser.asdl.asdl import Field, ASDLCompositeType

class AbstractSyntaxTree(object):
    def __init__(self, production, realized_fields=None, additional_info=None):
        self.production = production
        self.additional_info = additional_info

        self.fields = []
        self.parent_field = None
        self.created_time = 0

        if realized_fields:
            assert len(realized_fields) == len(self.production.fields)

            for field in realized_fields:
                self.add_child(field)
        else:
            for field in self.production.fields:
                self.add_child(RealizedField(field))

    def add_child(self, realized_field):
        self.fields.append(realized_field)
        realized_field.parent_node = self

    def __getitem__(self, field_name):
        for field in self.fields:
            if field.name == field_name:
                return field
        raise KeyError

    def sanity_check(self):
        if len(self.production.fields) != len(self.fields):
            raise ValueError('filed number must match')
        for field, realized_field in zip(self.production.fields, self.fields):
            assert field == realized_field.field
        for child in self.fields:
            for child_val in child.as_value_list:
                if isinstance(child_val, AbstractSyntaxTree):
                    child_val.sanity_check()

    def copy(self):
        new_tree = AbstractSyntaxTree(self.production)
        new_tree.created_time = self.created_time
        new_tree.additional_info = self.additional_info
        for i, old_field in enumerate(self.fields):
            new_field = new_tree.fields[i]
            new_field._not_single_cardinality_finished = old_field._not_single_cardinality_finished
            if isinstance(old_field.type, ASDLCompositeType):
                for value in old_field.as_value_list:
                    new_field.add_value(value.copy())
            else:
                for value in old_field.as_value_list:
                    new_field.add_value(value)

        return new_tree

    def to_string(self, sb=None):
        is_root = False
        if sb is None:
            is_root = True
            sb = StringIO()

        sb.write('(')
        sb.write(self.production.constructor.name)

        for field in self.fields:
            sb.write(' ')
            sb.write('(')
            sb.write(field.type.name)
            sb.write(Field.get_cardinality_repr(field.cardinality))
            sb.write('-')
            sb.write(field.name)

            if field.value is not None:
                for val_node in field.as_value_list:
                    sb.write(' ')
                    if isinstance(field.type, ASDLCompositeType):
                        val_node.to_string(sb)
                    else:
                        sb.write(str(val_node).replace(' ', '-SPACE-'))

            sb.write(')')

        sb.write(')')

        if is_root:
            return sb.getvalue()

    def __hash__(self):
        code = hash(self.production)
        for field in self.fields:
            code = code + 37 * hash(field)

        return code

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        if self.created_time != other.created_time:
            print(self, self.created_time)
            print(other, other.created_time)
            return False

        if self.production != other.production:
            return False

        if len(self.fields) != len(other.fields):
            return False

        for i in range(len(self.fields)):
            if self.fields[i] != other.fields[i]:
                return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return repr(self.production)

    @property
    def size(self):
        node_num = 1
        for field in self.fields:
            for val in field.as_value_list:
                if isinstance(val, AbstractSyntaxTree):
                    node_num += val.size
                else:
                    node_num += 1

        return node_num


class RealizedField(Field):
    def __init__(self, field, value=None, parent=None):
        super(RealizedField, self).__init__(field.name, field.type, field.cardinality)

        self.parent_node = None

        self.field = field

        if self.cardinality == 'multiple':
            self.value = []
            if value is not None:
                for child_node in value:
                    self.add_value(child_node)
        else:
            self.value = None
            if value is not None:
                self.add_value(value)

        self._not_single_cardinality_finished = False

    def add_value(self, value):
        if isinstance(value, AbstractSyntaxTree):
            value.parent_field = self

        if self.cardinality == 'multiple':
            self.value.append(value)
        else:
            self.value = value

    @property
    def as_value_list(self):
        if self.cardinality == 'multiple':
            return self.value
        elif self.value is not None:
            return [self.value]
        else:
            return []

    @property
    def finished(self):
        if self.cardinality == 'single':
            if self.value is None:
                return False
            else:
                return True
        elif self.cardinality == 'optional' and self.value is not None:
            return True
        else:
            if self._not_single_cardinality_finished:
                return True
            else:
                return False

    def set_finish(self):
        self._not_single_cardinality_finished = True

    def __eq__(self, other):
        if super(RealizedField, self).__eq__(other):
            if type(other) == Field:
                return True
            if self.value == other.value:
                return True
            else:
                return False
        else:
            return False
