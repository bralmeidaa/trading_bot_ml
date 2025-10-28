import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Clock, 
  Plus, 
  Edit3, 
  Trash2, 
  AlertTriangle, 
  CheckCircle, 
  XCircle,
  Calendar,
  Shield,
  Info
} from 'lucide-react';
import { useTradingRestrictions, useCreateRestriction, useUpdateRestriction, useDeleteRestriction } from '../services/apiClient';
import toast from 'react-hot-toast';

const DAYS_OF_WEEK = [
  { value: 0, label: 'Domingo', short: 'Dom' },
  { value: 1, label: 'Segunda', short: 'Seg' },
  { value: 2, label: 'Terça', short: 'Ter' },
  { value: 3, label: 'Quarta', short: 'Qua' },
  { value: 4, label: 'Quinta', short: 'Qui' },
  { value: 5, label: 'Sexta', short: 'Sex' },
  { value: 6, label: 'Sábado', short: 'Sáb' }
];

const RestrictionCard = ({ restriction, onEdit, onDelete }) => {
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  const handleDelete = () => {
    onDelete(restriction.id);
    setShowDeleteConfirm(false);
  };

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 shadow-sm hover:shadow-md transition-shadow"
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className={`p-2 rounded-lg ${restriction.is_active ? 'bg-red-100 dark:bg-red-900/20' : 'bg-gray-100 dark:bg-gray-700'}`}>
            <Shield className={`w-5 h-5 ${restriction.is_active ? 'text-red-600 dark:text-red-400' : 'text-gray-500'}`} />
          </div>
          <div>
            <h3 className="font-semibold text-gray-900 dark:text-white">{restriction.name}</h3>
            <p className="text-sm text-gray-500 dark:text-gray-400">{restriction.description}</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <span className={`px-2 py-1 rounded-full text-xs font-medium ${
            restriction.is_active 
              ? 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400' 
              : 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'
          }`}>
            {restriction.is_active ? 'Ativa' : 'Inativa'}
          </span>
        </div>
      </div>

      <div className="space-y-3">
        <div className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-300">
          <Calendar className="w-4 h-4" />
          <span>{restriction.formatted_period}</span>
        </div>
        
        <div className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-300">
          <Clock className="w-4 h-4" />
          <span>Timezone: {restriction.timezone}</span>
        </div>
      </div>

      <div className="flex items-center justify-between mt-6 pt-4 border-t border-gray-200 dark:border-gray-700">
        <div className="text-xs text-gray-500 dark:text-gray-400">
          Criada em {new Date(restriction.created_at).toLocaleDateString('pt-BR')}
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={() => onEdit(restriction)}
            className="p-2 text-gray-500 hover:text-blue-600 dark:text-gray-400 dark:hover:text-blue-400 transition-colors"
            title="Editar restrição"
          >
            <Edit3 className="w-4 h-4" />
          </button>
          
          <button
            onClick={() => setShowDeleteConfirm(true)}
            className="p-2 text-gray-500 hover:text-red-600 dark:text-gray-400 dark:hover:text-red-400 transition-colors"
            title="Deletar restrição"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      <AnimatePresence>
        {showDeleteConfirm && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
            onClick={() => setShowDeleteConfirm(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-md mx-4"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center space-x-3 mb-4">
                <AlertTriangle className="w-6 h-6 text-red-600" />
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Confirmar Exclusão
                </h3>
              </div>
              
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                Tem certeza que deseja deletar a restrição "{restriction.name}"? 
                Esta ação não pode ser desfeita.
              </p>
              
              <div className="flex justify-end space-x-3">
                <button
                  onClick={() => setShowDeleteConfirm(false)}
                  className="px-4 py-2 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                >
                  Cancelar
                </button>
                <button
                  onClick={handleDelete}
                  className="px-4 py-2 bg-red-600 text-white hover:bg-red-700 rounded-lg transition-colors"
                >
                  Deletar
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

const RestrictionForm = ({ restriction, onSave, onCancel }) => {
  const [formData, setFormData] = useState({
    name: restriction?.name || '',
    description: restriction?.description || '',
    start_day_of_week: restriction?.start_day_of_week ?? 6, // Default: Saturday
    start_time: restriction?.start_time || '07:00:00',
    end_day_of_week: restriction?.end_day_of_week ?? 0, // Default: Sunday
    end_time: restriction?.end_time || '20:00:00',
    timezone: restriction?.timezone || 'America/Sao_Paulo',
    is_active: restriction?.is_active ?? true
  });

  const [errors, setErrors] = useState({});

  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.name.trim()) {
      newErrors.name = 'Nome é obrigatório';
    }
    
    if (!formData.start_time) {
      newErrors.start_time = 'Horário inicial é obrigatório';
    }
    
    if (!formData.end_time) {
      newErrors.end_time = 'Horário final é obrigatório';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (validateForm()) {
      onSave(formData);
    }
  };

  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: null }));
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
      onClick={onCancel}
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
            {restriction ? 'Editar Restrição' : 'Nova Restrição'}
          </h2>
          <button
            onClick={onCancel}
            className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
          >
            <XCircle className="w-6 h-6" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Nome e Descrição */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Nome da Restrição *
              </label>
              <input
                type="text"
                value={formData.name}
                onChange={(e) => handleChange('name', e.target.value)}
                className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white ${
                  errors.name ? 'border-red-500' : 'border-gray-300'
                }`}
                placeholder="Ex: Restrição de Final de Semana"
              />
              {errors.name && <p className="text-red-500 text-sm mt-1">{errors.name}</p>}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Timezone
              </label>
              <select
                value={formData.timezone}
                onChange={(e) => handleChange('timezone', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
              >
                <option value="America/Sao_Paulo">America/Sao_Paulo</option>
                <option value="UTC">UTC</option>
                <option value="America/New_York">America/New_York</option>
                <option value="Europe/London">Europe/London</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Descrição
            </label>
            <textarea
              value={formData.description}
              onChange={(e) => handleChange('description', e.target.value)}
              rows={3}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
              placeholder="Descreva o motivo da restrição..."
            />
          </div>

          {/* Período de Restrição */}
          <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
              Período de Restrição
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Início */}
              <div>
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Início</h4>
                <div className="space-y-3">
                  <div>
                    <label className="block text-xs text-gray-600 dark:text-gray-400 mb-1">Dia da Semana</label>
                    <select
                      value={formData.start_day_of_week}
                      onChange={(e) => handleChange('start_day_of_week', parseInt(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                    >
                      {DAYS_OF_WEEK.map(day => (
                        <option key={day.value} value={day.value}>{day.label}</option>
                      ))}
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-xs text-gray-600 dark:text-gray-400 mb-1">Horário</label>
                    <input
                      type="time"
                      value={formData.start_time.slice(0, 5)} // Remove seconds for display
                      onChange={(e) => handleChange('start_time', e.target.value + ':00')}
                      className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white ${
                        errors.start_time ? 'border-red-500' : 'border-gray-300'
                      }`}
                    />
                    {errors.start_time && <p className="text-red-500 text-xs mt-1">{errors.start_time}</p>}
                  </div>
                </div>
              </div>

              {/* Fim */}
              <div>
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Fim</h4>
                <div className="space-y-3">
                  <div>
                    <label className="block text-xs text-gray-600 dark:text-gray-400 mb-1">Dia da Semana</label>
                    <select
                      value={formData.end_day_of_week}
                      onChange={(e) => handleChange('end_day_of_week', parseInt(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                    >
                      {DAYS_OF_WEEK.map(day => (
                        <option key={day.value} value={day.value}>{day.label}</option>
                      ))}
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-xs text-gray-600 dark:text-gray-400 mb-1">Horário</label>
                    <input
                      type="time"
                      value={formData.end_time.slice(0, 5)} // Remove seconds for display
                      onChange={(e) => handleChange('end_time', e.target.value + ':00')}
                      className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white ${
                        errors.end_time ? 'border-red-500' : 'border-gray-300'
                      }`}
                    />
                    {errors.end_time && <p className="text-red-500 text-xs mt-1">{errors.end_time}</p>}
                  </div>
                </div>
              </div>
            </div>

            {/* Preview */}
            <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <div className="flex items-center space-x-2 text-sm text-blue-800 dark:text-blue-200">
                <Info className="w-4 h-4" />
                <span>
                  Período: {DAYS_OF_WEEK[formData.start_day_of_week]?.label} {formData.start_time.slice(0, 5)} 
                  {formData.start_day_of_week !== formData.end_day_of_week ? 
                    ` até ${DAYS_OF_WEEK[formData.end_day_of_week]?.label} ${formData.end_time.slice(0, 5)}` :
                    ` às ${formData.end_time.slice(0, 5)}`
                  }
                </span>
              </div>
            </div>
          </div>

          {/* Status */}
          <div className="flex items-center space-x-3">
            <input
              type="checkbox"
              id="is_active"
              checked={formData.is_active}
              onChange={(e) => handleChange('is_active', e.target.checked)}
              className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
            />
            <label htmlFor="is_active" className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Restrição ativa
            </label>
          </div>

          {/* Buttons */}
          <div className="flex justify-end space-x-3 pt-6 border-t border-gray-200 dark:border-gray-700">
            <button
              type="button"
              onClick={onCancel}
              className="px-4 py-2 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
            >
              Cancelar
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 text-white hover:bg-blue-700 rounded-lg transition-colors"
            >
              {restriction ? 'Atualizar' : 'Criar'} Restrição
            </button>
          </div>
        </form>
      </motion.div>
    </motion.div>
  );
};

export default function TradingRestrictions() {
  const [showForm, setShowForm] = useState(false);
  const [editingRestriction, setEditingRestriction] = useState(null);

  const { data: restrictions = [], isLoading, refetch } = useTradingRestrictions();
  const createMutation = useCreateRestriction();
  const updateMutation = useUpdateRestriction();
  const deleteMutation = useDeleteRestriction();

  const handleCreate = () => {
    setEditingRestriction(null);
    setShowForm(true);
  };

  const handleEdit = (restriction) => {
    setEditingRestriction(restriction);
    setShowForm(true);
  };

  const handleSave = async (formData) => {
    try {
      if (editingRestriction) {
        await updateMutation.mutateAsync({
          id: editingRestriction.id,
          data: formData
        });
        toast.success('Restrição atualizada com sucesso!');
      } else {
        await createMutation.mutateAsync(formData);
        toast.success('Restrição criada com sucesso!');
      }
      setShowForm(false);
      setEditingRestriction(null);
      refetch();
    } catch (error) {
      toast.error(`Erro ao ${editingRestriction ? 'atualizar' : 'criar'} restrição: ${error.message}`);
    }
  };

  const handleDelete = async (restrictionId) => {
    try {
      await deleteMutation.mutateAsync(restrictionId);
      toast.success('Restrição deletada com sucesso!');
      refetch();
    } catch (error) {
      toast.error(`Erro ao deletar restrição: ${error.message}`);
    }
  };

  const activeRestrictions = restrictions.filter(r => r.is_active);
  const inactiveRestrictions = restrictions.filter(r => !r.is_active);

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Restrições de Horário</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {[1, 2, 3].map(i => (
            <div key={i} className="bg-gray-200 dark:bg-gray-700 rounded-lg h-48 animate-pulse" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
            Restrições de Horário
          </h2>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Configure períodos em que o trading deve ser bloqueado para evitar instabilidades
          </p>
        </div>
        
        <button
          onClick={handleCreate}
          className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white hover:bg-blue-700 rounded-lg transition-colors"
        >
          <Plus className="w-4 h-4" />
          <span>Nova Restrição</span>
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-red-100 dark:bg-red-900/20 rounded-lg">
              <Shield className="w-5 h-5 text-red-600 dark:text-red-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">{activeRestrictions.length}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">Restrições Ativas</p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-gray-100 dark:bg-gray-700 rounded-lg">
              <Shield className="w-5 h-5 text-gray-500" />
            </div>
            <div>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">{inactiveRestrictions.length}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">Restrições Inativas</p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-100 dark:bg-blue-900/20 rounded-lg">
              <Clock className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">{restrictions.length}</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">Total de Restrições</p>
            </div>
          </div>
        </div>
      </div>

      {/* Restrictions List */}
      {restrictions.length === 0 ? (
        <div className="text-center py-12">
          <Shield className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            Nenhuma restrição configurada
          </h3>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            Crie restrições de horário para bloquear operações em períodos instáveis
          </p>
          <button
            onClick={handleCreate}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white hover:bg-blue-700 rounded-lg transition-colors mx-auto"
          >
            <Plus className="w-4 h-4" />
            <span>Criar Primeira Restrição</span>
          </button>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Active Restrictions */}
          {activeRestrictions.length > 0 && (
            <div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4 flex items-center space-x-2">
                <CheckCircle className="w-5 h-5 text-red-600" />
                <span>Restrições Ativas ({activeRestrictions.length})</span>
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <AnimatePresence>
                  {activeRestrictions.map(restriction => (
                    <RestrictionCard
                      key={restriction.id}
                      restriction={restriction}
                      onEdit={handleEdit}
                      onDelete={handleDelete}
                    />
                  ))}
                </AnimatePresence>
              </div>
            </div>
          )}

          {/* Inactive Restrictions */}
          {inactiveRestrictions.length > 0 && (
            <div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4 flex items-center space-x-2">
                <XCircle className="w-5 h-5 text-gray-500" />
                <span>Restrições Inativas ({inactiveRestrictions.length})</span>
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <AnimatePresence>
                  {inactiveRestrictions.map(restriction => (
                    <RestrictionCard
                      key={restriction.id}
                      restriction={restriction}
                      onEdit={handleEdit}
                      onDelete={handleDelete}
                    />
                  ))}
                </AnimatePresence>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Form Modal */}
      <AnimatePresence>
        {showForm && (
          <RestrictionForm
            restriction={editingRestriction}
            onSave={handleSave}
            onCancel={() => {
              setShowForm(false);
              setEditingRestriction(null);
            }}
          />
        )}
      </AnimatePresence>
    </div>
  );
}